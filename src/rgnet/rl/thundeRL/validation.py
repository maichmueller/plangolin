import itertools
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import pymimir as mi
import torch
from lightning import Callback
from lightning.pytorch.core.hooks import ModelHooks
from tensordict import NestedKey, TensorDict
from torchrl.modules import ValueOperator

from rgnet.rl import ActorCritic
from rgnet.rl.envs.planning_env import PlanningEnvironment


def optimal_policy(space: mi.StateSpace) -> Dict[int, Set[int]]:
    # index of state to set of indices of optimal actions
    # optimal[i] = {j},  0 <= j < len(space.get_forward_transitions(space.get_states()[i]))
    optimal: Dict[int, Set[int]] = dict()
    for i, state in enumerate(space.get_states()):
        best_distance = min(
            space.get_distance_to_goal_state(t.target)
            for t in space.get_forward_transitions(state)
        )
        best_actions: Set[int] = set(
            idx
            for idx, t in enumerate(space.get_forward_transitions(state))
            if space.get_distance_to_goal_state(t.target) == best_distance
        )
        optimal[i] = best_actions
    return optimal


class CriticValidation(torch.nn.Module, Callback):

    def __init__(
        self,
        optimal_values_dict: Dict[int, torch.Tensor],
        value_operator: ValueOperator,
        loss_function=torch.nn.functional.mse_loss,
        log_name: str = "value_loss",
    ):
        super().__init__()
        for space_idx, optimal_values in optimal_values_dict.items():
            self.register_buffer(str(space_idx), optimal_values)
        self.value_op = value_operator
        self.loss_function = loss_function
        self.log_name: str = log_name
        self.epoch_values = defaultdict(list)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_values.clear()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for dataloader_idx, values in self.epoch_values.items():
            epoch_mean = sum(values) / len(values)
            pl_module.log(
                f"val/{self.log_name}_{dataloader_idx}", epoch_mean, on_epoch=True
            )

    def forward(self, tensordict: TensorDict, dataloader_idx=0):

        try:
            optimal_values: torch.Tensor = self.get_buffer(str(dataloader_idx))
        except AttributeError:
            warnings.warn(
                f"No optimal values found for dataloader_idx {dataloader_idx}"
            )
            return

        prediction = self.value_op(tensordict).squeeze(dim=-1)
        state_value: torch.Tensor = prediction[ActorCritic.default_keys.state_value]

        if optimal_values.device != state_value.device:
            warnings.warn(
                f"Found missmatching devices: {optimal_values.device=} and { state_value.device=}"
            )
        # shape is batch_size
        indices: torch.Tensor = tensordict["idx_in_space"]
        target = optimal_values[indices]

        loss = self.loss_function(state_value, target)
        self.epoch_values[dataloader_idx].append(loss.item())


class PolicyValidation(torch.nn.Module, Callback):

    def __init__(
        self,
        optimal: Dict[int, Dict[int, Set[int]]],
        keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
        log_name: str = "policy_precision",
    ) -> None:
        super().__init__()
        # Outer dictionary maps datalaoder_idx to the respective StateSpace
        # Inner dict maps from state index to index of best target states
        self.optimal_action_indices: Dict[int, Dict[int, Set[int]]] = optimal
        self.keys = keys
        self.log_name = log_name

        self.epoch_values = defaultdict(list)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_values.clear()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for dataloader_idx, values in self.epoch_values.items():
            epoch_mean = sum(values) / len(values)
            pl_module.log(
                f"val/{self.log_name}_{dataloader_idx}", epoch_mean, on_epoch=True
            )

    def forward(self, tensordict: TensorDict, dataloader_idx=0):
        """
        Assesses the policy as number of states in which an optimal action was chosen
        divided by the number of states.
        Two alternatives could be considered too:
         - Factor in the number of transitions as denominator
         - Use the actual probabilities and not just the sampled action
        :param tensordict: Result of agent on environment. Specifically we expect
            - "idx_in_space" identifying the current state in the respective StateSpace
            - PlanningEnvironment.keys.action the idx of the transition which the agent chose
        :param dataloader_idx: Each dataloader is associated with a specific StateSpace.
        :return: Dictionary of metric_name: policy_precision
        """
        # TODO use cross entropy loss between all optimal actions and policy probs

        # Trigger cpu synchronisation by tolist()
        # Tensordict has additional time dimension
        assert tensordict.names[-1] == "time" and tensordict.batch_size[-1] == 1
        # len() = batch_size
        state_indices: List[int] = tensordict["idx_in_space"].squeeze().tolist()
        action_indices: List[int] = tensordict[self.keys.action].squeeze().tolist()
        if dataloader_idx not in self.optimal_action_indices:
            warnings.warn(
                f"No optimal actions found for dataloader_idx {dataloader_idx}"
            )
            return
        optimal_actions = self.optimal_action_indices[dataloader_idx]
        correct_actions = 0
        for action_idx, state_idx in zip(action_indices, state_indices):
            if action_idx in optimal_actions[state_idx]:
                correct_actions += 1
        self.epoch_values[dataloader_idx].append(correct_actions / len(state_indices))


class MetricsHook(torch.nn.Module, Callback):

    def __init__(
        self,
        env_keys: PlanningEnvironment.AcceptedKeys,
        probs_key: NestedKey,
        save_file: Path,
    ):
        super().__init__()
        self.env_keys = env_keys
        self.probs_key: NestedKey = probs_key
        assert isinstance(save_file, Path)
        self.save_file: Path = save_file
        self.epoch = 0
        # TODO Extend to multiple dataloader
        # Collected over one validation epoch
        # The index of the current states in their StateSpaces collected over one epoch
        self.state_id_in_epoch: List[torch.Tensor] = list()
        # The probability for each outgoing transition over one epoch
        # Each state can have various number of successor therefore we have a list of list
        self.probs_in_epoch: List[List[torch.Tensor]] = list()

    def get_extra_state(self) -> Any:
        return {
            "env_keys": self.env_keys,
            "probs_key": self.probs_key,
            "save_file": self.save_file,
            "epoch": self.epoch,
        }

    def set_extra_state(self, state: Any) -> None:
        self.env_keys = state["env_keys"]
        self.probs_key = state["probs_key"]
        self.save_file = state["save_file"]
        self.epoch = state["epoch"]

    def forward(self, tensordict: TensorDict, dataloader_idx=0, **kwargs):
        if dataloader_idx != 0:
            warnings.warn(f"Only implemented for single dataloader {dataloader_idx}")
            return {}
        # We have the additional time dimension (also for now only one step)
        batched_probs: List[List[torch.Tensor]] = tensordict[self.probs_key]
        batched_probs: List[torch.Tensor] = [ls[0].detach() for ls in batched_probs]
        state_indices = tensordict["idx_in_space"].squeeze()
        self.state_id_in_epoch.append(state_indices)
        self.probs_in_epoch.append(batched_probs)
        return

    def on_validation_epoch_start(self, trainer, module) -> None:
        self.state_id_in_epoch.clear()
        self.probs_in_epoch.clear()

    def on_validation_epoch_end(self, trainer, module) -> None:
        flattened_indices = torch.cat(self.state_id_in_epoch)
        sorted_ids, new_indices = torch.sort(flattened_indices)
        new_indices_list = new_indices.tolist()
        flattened_probs: List[torch.Tensor] = list(
            itertools.chain.from_iterable(self.probs_in_epoch)
        )
        sorted_probs: List[torch.Tensor] = []
        for i in new_indices_list:
            sorted_probs.append(flattened_probs[i])

        file_name = self.save_file.parent / (
            self.save_file.stem + str(self.epoch) + ".pt"
        )
        torch.save(sorted_probs, file_name)
        self.epoch += 1
