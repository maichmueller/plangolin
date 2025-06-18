import dataclasses
import itertools
from typing import Any, Iterator, List, Optional, Tuple, Union

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import LazyStackedTensorDict, TensorDict
from torch.nn import ModuleList
from torch_geometric.data import Batch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from rgnet.models.pyg_module import PyGHeteroModule, PyGModule
from rgnet.rl.agents import ActorCritic
from rgnet.rl.envs import PlanningEnvironment
from rgnet.rl.losses import AllActionsLoss, CriticLoss
from rgnet.rl.thundeRL.validation import ValidationCallback
from rgnet.utils.misc import as_non_tensor_stack
from rgnet.utils.object_embeddings import ObjectEmbedding


class PolicyGradientLitModule(lightning.LightningModule):
    @dataclasses.dataclass
    class _AcceptedKeys:
        all_rewards: str = "all_rewards"
        all_dones: str = "all_dones"
        idx_in_space: str = "idx_in_space"

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        gnn: Union[PyGModule, PyGHeteroModule],
        actor_critic: ActorCritic,
        loss: CriticLoss,
        optim: torch.optim.Optimizer,
        validation_hooks: Optional[List[ValidationCallback]] = None,
        add_all_rewards_and_done: bool = False,
        add_successor_embeddings: bool = False,
        keys: _AcceptedKeys = default_keys,
    ) -> None:
        super().__init__()
        assert isinstance(actor_critic, ActorCritic)
        assert isinstance(loss, CriticLoss)
        assert isinstance(optim, torch.optim.Optimizer)
        if not isinstance(gnn, PyGHeteroModule) and not isinstance(gnn, PyGModule):
            raise ValueError(f"Unknown GNN type: {gnn}")
        self.gnn = gnn
        self.actor_critic = actor_critic
        self.loss = loss
        self.optim = optim
        self.validation_hooks = ModuleList(validation_hooks or [])
        is_all_actions = isinstance(loss, AllActionsLoss)
        self.add_all_rewards_and_done = add_all_rewards_and_done or is_all_actions
        self.add_successor_embeddings = add_successor_embeddings or is_all_actions
        self.keys = keys
        self._cuda_streams = None
        self._cuda_cycler: Iterator[torch.cuda.Stream | None] = None

    def next_stream(self) -> Optional[torch.cuda.Stream]:
        """Returns the next CUDA stream if available, otherwise None."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            if self._cuda_streams is None:
                self._cuda_streams = [torch.cuda.Stream(self.device) for _ in range(2)]
                self._cuda_cycler = itertools.cycle(self._cuda_streams)
            return next(self._cuda_cycler)
        return None

    @property
    def cuda_streams(self) -> Optional[List[torch.cuda.Stream]]:
        """Returns the list of CUDA streams if available, otherwise None."""
        return self._cuda_streams

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

    def on_fit_start(self):
        # pass the device to the DataModule
        self.trainer.datamodule.device = self.device

    def forward(
        self,
        states_data: Batch,
        successors_flattened: Batch,
        num_successors: torch.Tensor,
    ) -> TensorDict:
        # Required to group the flattened tensors by each state.
        # E.g., group all the embeddings for each successor state
        successor_start_indices = num_successors.cumsum(dim=0).long()[:-1]
        successor_start_indices_cpu = successor_start_indices.cpu()

        batches_out: list[ObjectEmbedding] = []
        for batch in (states_data, successors_flattened):
            with torch.cuda.stream(self.next_stream()):
                # Shape batch_size x embedding_size
                batches_out.append(ObjectEmbedding.from_sparse(*self.gnn(batch)))
        if self.cuda_streams is not None:
            for stream in self.cuda_streams:
                stream.synchronize()
        object_embeddings, successor_embeddings = batches_out
        # Sample actions from the agent
        batched_probs: List[torch.Tensor]  # probability for each transition
        batched_probs, action_indices, log_probs = self.actor_critic.embedded_forward(
            object_embeddings, successor_embeddings, num_successors=num_successors
        )
        # We map the action_indices which are per state into the successor tensor
        # which contains the concatenation of all successor object embeddings.
        # successor_start_indices[i] is the position where the successors of state i start.
        # so successor_start_indices[i] + actions_indices[i] is the i-th successor in the continuous successor tensor.
        successor_action_indices = torch.cat(
            [
                torch.tensor([0], device=successor_start_indices.device),
                successor_start_indices,
            ]
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

        ac_keys = self.actor_critic.keys
        # TODO: can we decouple this through an argument as well or is this class tightly coupled with PlanningEnvironment?
        #  if so, we should clarify this somewhere (ie docstring or class name etc)
        env_keys = PlanningEnvironment.default_keys
        # Data corresponding to the next state -> the result of applying the actions
        next_td = TensorDict(
            {
                ac_keys.current_embedding: next_object_embeddings.to_tensordict(),
                env_keys.reward: rewards,
                env_keys.done: terminated,
                env_keys.terminated: terminated,
            },
            batch_size=(states_data.batch_size,),
        )

        td_dict = {
            env_keys.action: action_indices,
            ac_keys.current_embedding: object_embeddings.to_tensordict(),
            ac_keys.log_probs: log_probs,
            ac_keys.probs: as_non_tensor_stack(batched_probs),
            self.keys.idx_in_space: states_data.idx,  # torch.Tensor int64
            "next": next_td,
        }
        if self.add_all_rewards_and_done:
            all_rewards: List[torch.Tensor] = list(
                states_data.reward.tensor_split(successor_start_indices_cpu)
            )
            all_dones: List[torch.Tensor] = list(
                states_data.done.tensor_split(successor_start_indices_cpu)
            )
            td_dict |= {
                self.keys.all_rewards: as_non_tensor_stack(all_rewards),
                self.keys.all_dones: as_non_tensor_stack(all_dones),
            }
        if self.add_successor_embeddings:
            split_embeddings: List[ObjectEmbedding] = successor_embeddings.tensor_split(
                successor_start_indices_cpu
            )
            td_dict[self.actor_critic.keys.successor_embeddings] = as_non_tensor_stack(
                map(lambda s: s.to_tensordict(), split_embeddings)
            )

        td = TensorDict(
            source=td_dict,
            batch_size=(states_data.batch_size,),
        )
        # Most losses / value estimators expect a time dimension.
        # Therefore, we create a "rollout" of shape batch_size x time a.k.a batch_size x 1.
        stacked = LazyStackedTensorDict.maybe_dense_stack([td], len(td.batch_size))
        stacked.refine_names(..., "time")
        return stacked

    @staticmethod
    def _loss_filter(key: str, value):
        return value if key.startswith("loss_") else None

    def _common_step(
        self, batch_tuple: Tuple[Batch, Batch, torch.Tensor, dict[str, Any]]
    ) -> tuple[TensorDict, torch.Tensor]:
        out: TensorDict = self(*batch_tuple[:-1])
        losses_td = self.loss(out)
        losses = losses_td.named_apply(self._loss_filter)
        return losses, losses.sum(reduce=True)

    def training_step(
        self, batch_tuple: Tuple[Batch, Batch, torch.Tensor, dict[str, Any]]
    ) -> STEP_OUTPUT:
        with set_exploration_type(ExplorationType.RANDOM):
            losses_separate, loss = self._common_step(batch_tuple)
            for key, value in losses_separate.items():
                self.log(
                    "train/" + key, value, batch_size=batch_tuple[-1]["batch_size"]
                )
        return loss

    def validation_step(
        self,
        batch_tuple: Tuple[Batch, Batch, torch.Tensor, dict[str, Any]],
        batch_idx=None,
        dataloader_idx=0,
    ):
        with set_exploration_type(ExplorationType.MODE):
            as_tensordict = self(*batch_tuple[:-1])
            for hook in self.validation_hooks:
                optional_metrics = hook(
                    as_tensordict, batch_idx=batch_idx, dataloader_idx=dataloader_idx
                )
                if optional_metrics is None:
                    continue
                for key, value in optional_metrics.items():
                    self.log(
                        "validation/" + key,
                        value,
                        batch_size=batch_tuple[-1]["batch_size"],
                    )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim
