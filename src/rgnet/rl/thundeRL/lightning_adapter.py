from typing import Iterable, List, Optional, Tuple

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import LazyStackedTensorDict, TensorDict
from torch.nn import ModuleList
from torch_geometric.data import Batch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import LossModule

from rgnet import HeteroGNN
from rgnet.rl.agents import ActorCritic
from rgnet.rl.envs import PlanningEnvironment
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack


class LightningAdapter(lightning.LightningModule):

    def __init__(
        self,
        gnn: HeteroGNN,
        actor_critic: ActorCritic,
        loss: LossModule,
        optim: torch.optim.Optimizer,
        validation_hooks: Optional[Iterable[torch.nn.Module]] = None,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        assert isinstance(gnn, HeteroGNN)
        assert isinstance(actor_critic, ActorCritic)
        assert isinstance(loss, LossModule)
        assert isinstance(optim, torch.optim.Optimizer)
        self.gnn = gnn
        self.actor_critic = actor_critic
        self.loss = loss
        self.optim = optim
        self.validation_hooks = ModuleList(validation_hooks or [])

    def _compute_embeddings(
        self,
        states_data: Batch,
        successors_flattened: Batch,
        slices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        slices_cpu = slices.cpu()
        # Shape batch_size x embedding_size
        state_embedding: torch.Tensor = self.gnn(
            states_data.x_dict, states_data.edge_index_dict, states_data.batch_dict
        )
        # shape (batch_size * num_successor[i]) x embedding_size
        successor_embedding: torch.Tensor = self.gnn(
            successors_flattened.x_dict,
            successors_flattened.edge_index_dict,
            successors_flattened.batch_dict,
        )

        batched_successor_embeddings: Tuple[torch.Tensor, ...] = (
            successor_embedding.tensor_split(slices_cpu)
        )
        return state_embedding, batched_successor_embeddings

    @staticmethod
    def _get_rewards_and_done(
        action_indices: torch.Tensor, slices: torch.Tensor, states_data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # reward and done are flattened 1D tensors
        # We map the action_indices which are per state into the continuous 1D tensor.
        # Slices are the indices where the flattened tensors have to be split.
        # so slices[i] + actions_indices[i] is the element in the 1D space
        flattened_action_indices = torch.cat(
            [torch.tensor([0], device=slices.device), slices]
        )
        flattened_action_indices = flattened_action_indices + action_indices
        rewards: torch.Tensor = states_data.reward[flattened_action_indices]
        terminated: torch.Tensor = states_data.done[flattened_action_indices]
        # unsqueeze to fit shape of value-operator output
        rewards = rewards.unsqueeze(dim=-1)
        terminated = terminated.unsqueeze(dim=-1)

        return rewards, terminated

    def forward(
        self,
        states_data: Batch,
        successors_flattened: Batch,
        num_successors: torch.Tensor,
    ) -> TensorDict:
        # Required to group the flattened tensors by each state
        # E.g. group all the embeddings for each successor state
        slices = num_successors.cumsum(dim=0).long()[:-1]

        state_embedding, batched_successor_embeddings = self._compute_embeddings(
            states_data, successors_flattened, slices
        )

        # Sample actions from the agent
        batched_probs: List[torch.Tensor]  # probability for each transition
        batched_probs, action_indices, log_probs = self.actor_critic.embedded_forward(
            state_embedding, batched_successor_embeddings
        )
        # Select the rewards for the chosen actions
        rewards, terminated = self._get_rewards_and_done(
            action_indices, slices, states_data
        )

        # Select the next-states by the action index chosen for each batch entry
        next_states = torch.stack(
            [
                batched_successor[action_idx]
                for action_idx, batched_successor in zip(
                    action_indices, batched_successor_embeddings
                )
            ]
        )

        # Data corresponding to the next state -> the result of applying the actions
        next_td = TensorDict(
            {
                ActorCritic.default_keys.current_embedding: next_states,
                PlanningEnvironment.default_keys.reward: rewards,
                PlanningEnvironment.default_keys.done: terminated,
                PlanningEnvironment.default_keys.terminated: terminated,
            },
            batch_size=(states_data.batch_size,),
        )

        td = TensorDict(
            {
                PlanningEnvironment.default_keys.action: action_indices,
                ActorCritic.default_keys.current_embedding: state_embedding,
                self.actor_critic.keys.log_probs: log_probs,
                self.actor_critic.keys.probs: as_non_tensor_stack(batched_probs),
                "idx_in_space": states_data.idx,  # torch.Tensor int64
                "next": next_td,
            },
            batch_size=(states_data.batch_size,),
        )
        # Most losses / value estimators expect a time dimension
        # Therefore we create a "rollout" of shape batch_size x time a.k.a batch_size x 1
        stacked = LazyStackedTensorDict.maybe_dense_stack([td], len(td.batch_size))
        stacked.refine_names(..., "time")
        return stacked

    def _apply_loss(self, td: TensorDict) -> tuple[TensorDict, torch.Tensor]:
        # Apply the loss and return the loss components and combined loss
        losses_td = self.loss(td)
        losses = losses_td.named_apply(
            lambda key, value: value if key.startswith("loss_") else None
        )
        return losses, losses.sum(reduce=True)

    def _common_step(self, batch_tuple: Tuple[Batch, Batch, torch.Tensor]):
        out: TensorDict = self(*batch_tuple)
        return self._apply_loss(out)

    def training_step(
        self, batch_tuple: Tuple[Batch, Batch, torch.Tensor]
    ) -> STEP_OUTPUT:
        with set_exploration_type(ExplorationType.RANDOM):
            losses_separate, loss = self._common_step(batch_tuple)
            for key, value in losses_separate.items():
                self.log("train/" + key, value, batch_size=batch_tuple[0].batch_size)
        return loss

    def validation_step(
        self,
        batch_tuple: Tuple[Batch, Batch, torch.Tensor],
        batch_idx=None,
        dataloader_idx=0,
    ):
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            as_tensordict = self.forward(*batch_tuple)
            for hook in self.validation_hooks:
                optional_metrics = hook(as_tensordict, dataloader_idx=dataloader_idx)
                if optional_metrics is None:
                    continue
                for key, value in optional_metrics.items():
                    self.log("val/" + key, value, batch_size=batch_tuple[0].batch_size)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim
