from typing import Dict, List, Optional, Tuple, Union

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import LazyStackedTensorDict, TensorDict
from torch.nn import ModuleList
from torch_geometric.data import Batch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import LossModule

from rgnet.models.pyg_module import PyGHeteroModule, PyGModule
from rgnet.rl.agents import ActorCritic
from rgnet.rl.envs import PlanningEnvironment
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack
from rgnet.rl.thundeRL.validation import ValidationCallback
from rgnet.utils.object_embeddings import ObjectEmbedding


class PolicyGradientModule(lightning.LightningModule):
    def __init__(
        self,
        gnn: Union[PyGModule, PyGHeteroModule],
        actor_critic: ActorCritic,
        loss: LossModule,
        optim: torch.optim.Optimizer,
        validation_hooks: Optional[List[ValidationCallback]] = None,
    ) -> None:
        super().__init__()
        assert isinstance(actor_critic, ActorCritic)
        assert isinstance(loss, LossModule)
        assert isinstance(optim, torch.optim.Optimizer)
        if not isinstance(gnn, PyGHeteroModule) and not isinstance(gnn, PyGModule):
            raise ValueError(f"Unknown GNN type: {gnn}")
        self.gnn = gnn
        self.actor_critic = actor_critic
        self.loss = loss
        self.optim = optim
        self.validation_hooks = ModuleList(validation_hooks or [])

    @staticmethod
    def _get_rewards_and_done(
        successor_action_indices: torch.Tensor, states_data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # reward and done are flattened 1D tensors
        rewards: torch.Tensor = states_data.reward[successor_action_indices]
        terminated: torch.Tensor = states_data.done[successor_action_indices]
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
        # Required to group the flattened tensors by each state.
        # E.g., group all the embeddings for each successor state
        slices = num_successors.cumsum(dim=0).long()[:-1]

        # Shape batch_size x embedding_size
        object_embeddings: ObjectEmbedding = self.gnn(states_data)
        # shape (batch_size * num_successor[i]) x embedding_size
        successor_embeddings: ObjectEmbedding = self.gnn(successors_flattened)

        # Sample actions from the agent
        batched_probs: List[torch.Tensor]  # probability for each transition
        batched_probs, action_indices, log_probs = self.actor_critic.embedded_forward(
            object_embeddings, successor_embeddings, num_successors=num_successors
        )
        # We map the action_indices which are per state into the successor tensor
        # which contains the concatenation of all successor object embeddings.
        # slices[i] is the position where the successors of state i start.
        # so slices[i] + actions_indices[i] is the i-th successor in the continuous successor tensor.
        successor_action_indices = torch.cat(
            [torch.tensor([0], device=slices.device), slices]
        )
        successor_action_indices = successor_action_indices + action_indices
        rewards, terminated = self._get_rewards_and_done(
            successor_action_indices, states_data
        )

        # Select the next-states by the action index chosen for each batch entry
        next_object_embeddings = ObjectEmbedding(
            successor_embeddings.dense_embedding[successor_action_indices],
            padding_mask=successor_embeddings.padding_mask[successor_action_indices],
        )

        # Data corresponding to the next state -> the result of applying the actions
        next_td = TensorDict(
            {
                ActorCritic.default_keys.current_embedding: next_object_embeddings.to_tensordict(),
                PlanningEnvironment.default_keys.reward: rewards,
                PlanningEnvironment.default_keys.done: terminated,
                PlanningEnvironment.default_keys.terminated: terminated,
            },
            batch_size=(states_data.batch_size,),
        )

        td = TensorDict(
            {
                PlanningEnvironment.default_keys.action: action_indices,
                ActorCritic.default_keys.current_embedding: object_embeddings.to_tensordict(),
                self.actor_critic.keys.log_probs: log_probs,
                self.actor_critic.keys.probs: as_non_tensor_stack(batched_probs),
                "idx_in_space": states_data.idx,  # torch.Tensor int64
                "next": next_td,
            },
            batch_size=(states_data.batch_size,),
        )
        # Most losses / value estimators expect a time dimension.
        # Therefore, we create a "rollout" of shape batch_size x time a.k.a batch_size x 1.
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
        with set_exploration_type(ExplorationType.MODE):
            as_tensordict = self.forward(*batch_tuple)
            for hook in self.validation_hooks:
                optional_metrics = hook(
                    as_tensordict, batch_idx=batch_idx, dataloader_idx=dataloader_idx
                )
                if optional_metrics is None:
                    continue
                for key, value in optional_metrics.items():
                    self.log(
                        "validation/" + key, value, batch_size=batch_tuple[0].batch_size
                    )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim
