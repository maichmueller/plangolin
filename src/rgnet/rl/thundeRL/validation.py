import itertools
import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set

import pymimir as mi
import torch
from lightning import Callback
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


class ValidationCallback(torch.nn.Module, Callback):

    def __init__(
        self,
        log_name: str,
        dataloader_names: Optional[Dict[int, str]] = None,
        only_run_for_dataloader: Optional[set[int]] = None,
        epoch_reduction: Literal["mean", "max", "min"] = "mean",
    ) -> None:
        """

        :param log_name: Under which name the metrics should be logged.
        :param dataloader_names: Mapping each data loader to a string. Will be used in the full logg-key.
            The full key will be val/{log_name}_{dataloader_names[idx]. If not specified the index
            of the dataloader will be used instead.
        :param only_run_for_dataloader: Optional parameter to limit the callback to a specific set of dataloader.
        :param epoch_reduction: How to reduce the values of one epoch for one dataloader.
            (default: mean)
        """
        super().__init__()
        self.log_name = log_name
        self.dataloader_names = dataloader_names
        self.only_run_for_dataloader = only_run_for_dataloader
        if epoch_reduction == "mean":
            self.epoch_reduction = statistics.fmean
        elif epoch_reduction == "max":
            self.epoch_reduction = max
        elif epoch_reduction == "min":
            self.epoch_reduction = min
        self.is_sanity_check: bool = False
        self.epoch_values = defaultdict(list)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_values.clear()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for dataloader_idx, values in self.epoch_values.items():
            pl_module.log(
                self.log_key(dataloader_idx),
                self.epoch_reduction(values),
                on_epoch=True,
            )

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.is_sanity_check = True

    def on_sanity_check_end(self, trainer, pl_module) -> None:
        self.is_sanity_check = False

    def log_key(self, dataloader_idx: int):
        return (
            f"val/{self.log_name}_" + self.dataloader_names[dataloader_idx]
            if self.dataloader_names
            else str(dataloader_idx)
        )

    def skip_dataloader(self, dataloader_index):
        return (
            self.only_run_for_dataloader is not None
            and dataloader_index not in self.only_run_for_dataloader
        )


class CriticValidation(ValidationCallback):

    def __init__(
        self,
        optimal_values_dict: Dict[int, torch.Tensor],
        value_operator: ValueOperator,
        loss_function: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        state_value_key: NestedKey = ActorCritic.default_keys.state_value,
        log_name: str = "value_loss",
        dataloader_names: Optional[Dict[int, str]] = None,
        only_run_for_dataloader: Optional[set[int]] = None,
    ):
        """
        Used to assess the quality of a value-operator that learns the distance to the goal state.
        Computes the current prediction of state values for current states and compares them
        to the optimal values using the loss function.
        This is a torch module in order to simplify moving the optimal values to the device.
        :param optimal_values_dict: Dictionary mapping the dataloader index to a flat tensor.
            optimal_values_dict[i][j] is the optimal value for the j-th state in the i-th validation space.
        :param value_operator: The value operator currently learned. Each in-key has to be contained in the tensordicts.
        :param loss_function: Which loss function to use. (default: torch.nn.functional.mse_loss)
        :param state_value_key: The output key of the value_operator.
        :param log_name: How should be logged. (default: value_loss)
        """
        super().__init__(
            log_name=log_name,
            dataloader_names=dataloader_names,
            only_run_for_dataloader=only_run_for_dataloader,
        )
        for space_idx, optimal_values in optimal_values_dict.items():
            self.register_buffer(str(space_idx), optimal_values)
        self.value_op = value_operator
        self.loss_function = loss_function or torch.nn.functional.mse_loss
        self.state_value_key = state_value_key

    def forward(self, tensordict: TensorDict, dataloader_idx=0):
        if self.skip_dataloader(dataloader_idx) or self.is_sanity_check:
            return
        try:
            optimal_values: torch.Tensor = self.get_buffer(str(dataloader_idx))
        except AttributeError:
            warnings.warn(
                f"No optimal values found for dataloader_idx {dataloader_idx}"
            )
            return

        # tensordict should be of the shape [batch_size, time, feature size]
        assert tensordict.batch_dims == 2
        assert tensordict.names[-1] == "time"

        # select(...) makes sure that the state_value is not written in the original.
        prediction: TensorDict = self.value_op(
            tensordict.select(*self.value_op.in_keys)
        )
        # tensordict has shape [batch_size, 1] and tensordict[current_embedding] is [batch_size, 1, hidden_size]
        # therefore, prediction has shape [batch_size, 1, 1] and we have to squeeze two dimensions.
        state_value: torch.Tensor = prediction[self.state_value_key].squeeze()

        if optimal_values.device != state_value.device:
            warnings.warn(
                f"Found mismatching devices: {optimal_values.device=} and { state_value.device=}"
            )
        # shape is batch_size
        indices: torch.Tensor = tensordict["idx_in_space"].squeeze(dim=-1)
        target = optimal_values[indices]

        loss = self.loss_function(state_value, target)
        self.epoch_values[dataloader_idx].append(loss.item())


class PolicyValidation(ValidationCallback):

    def __init__(
        self,
        optimal_policy_dict: Dict[int, Dict[int, Set[int]]],
        keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
        log_name: str = "policy_precision",
        dataloader_names: Optional[Dict[int, str]] = None,
        only_run_for_dataloader: Optional[set[int]] = None,
        epoch_reduction: Literal["mean", "max", "min"] = "mean",
    ) -> None:
        super().__init__(
            log_name=log_name,
            dataloader_names=dataloader_names,
            only_run_for_dataloader=only_run_for_dataloader,
            epoch_reduction=epoch_reduction,
        )
        # Outer dictionary maps dataloader_idx to the respective StateSpace
        # Inner dict maps from state index to index of best target states
        self.optimal_action_indices: Dict[int, Dict[int, Set[int]]] = (
            optimal_policy_dict
        )
        self.keys = keys

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
        """
        # TODO use cross entropy loss between all optimal actions and policy probs

        if self.skip_dataloader(dataloader_idx) or self.is_sanity_check:
            return

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


class PolicyEntropy(ValidationCallback):
    """
    Compute the entropy of the actor and aggregate over one validation epoch.
    The entropy is only computed for states with more than one successor, which
    is referred to as "non-trivial".
    Additionally, the entropy is normalized to be between 0 and 1. A learning actor will
    typically start with an entropy of 0 (close to a uniform distribution).
    This callback is especially useful to monitor the convergence of the actor.
    """

    def __init__(
        self,
        probs_key: NestedKey = ActorCritic.default_keys.probs,
        log_name: str = "non_trivial_policy_entropy",
        dataloader_names: Optional[Dict[int, str]] = None,
        only_run_for_dataloader: Optional[set[int]] = None,
        epoch_reduction: Literal["mean", "max", "min"] = "mean",
    ) -> None:
        super().__init__(
            log_name, dataloader_names, only_run_for_dataloader, epoch_reduction
        )
        self.probs_key = probs_key
        self.epoch_values = defaultdict(list)

    @staticmethod
    def normalized_entropy(probs_tensor: torch.Tensor):
        """
        Entropy over a tensor of a probability distribution.
        Normalized to [0,1], where 0 corresponds to a uniform distribution.
        :param probs_tensor: Tensor with sum = 1 and values between 0 and 1
        :return: Tensor of shape (1,) on the same device as the input tensor.
        """
        if probs_tensor.numel() == 1:
            return torch.zeros((1,), dtype=torch.float, device=probs_tensor.device)
        entropy = torch.distributions.Categorical(probs=probs_tensor).entropy()
        return entropy / torch.tensor(
            (probs_tensor.numel(),), device=probs_tensor.device
        )

    def forward(self, tensordict: TensorDict, dataloader_idx=0):
        if self.skip_dataloader(dataloader_idx):
            return
        batched_probs: List[List[torch.Tensor]] = tensordict[self.probs_key]
        batched_nontrivial_entropies = [
            PolicyEntropy.normalized_entropy(tensor_list[0])
            for tensor_list in batched_probs
            if tensor_list[0].numel() > 1
        ]
        if (
            len(batched_nontrivial_entropies) == 0
        ):  # All states in a batch could be filtered out
            return
        mean_entropy = torch.stack(batched_nontrivial_entropies).mean()
        self.epoch_values[dataloader_idx].append(mean_entropy)


class ProbsStoreCallback(ValidationCallback):

    def __init__(
        self,
        save_dir: Path,
        probs_key: NestedKey = ActorCritic.default_keys.probs,
        log_name: str = "actor_probs",
        dataloader_names: Optional[Dict[int, str]] = None,
        only_run_for_dataloader: Optional[set[int]] = None,
    ):
        """
        Save the probability for all successors for every state.
        The combined probabilities over one epoch are saved to a .pt file.
        The saved file is of the form List[torch.Tensor]. save_file[i][j] is the probability
        in state i for moving to the j-th successor.

        :param probs_key: Under which key the agent stores its probabilities. The
            probabilities are assumed to be of the form List[torch.Tensor]
        :param save_dir: Where to store the .pt files.
        :param log_name: Start of the file-names. (default: actor_probs)
        """
        super().__init__(
            log_name=log_name,
            dataloader_names=dataloader_names,
            only_run_for_dataloader=only_run_for_dataloader,
        )
        self.probs_key: NestedKey = probs_key
        assert isinstance(save_dir, Path)
        self.save_dir: Path = save_dir / self.log_name
        self.save_dir.mkdir(exist_ok=True)
        self.epoch: int = 0
        # Collected over one validation epoch
        # The index of the current states in their StateSpaces collected over one epoch
        self.state_id_in_epoch: Dict[int, List[torch.Tensor]] = defaultdict(list)
        # The probability for each outgoing transition over one epoch
        # Each state can have various number of successor therefore we have a list of list
        self.probs_in_epoch: Dict[int, List[List[torch.Tensor]]] = defaultdict(list)

    def get_extra_state(self) -> Any:
        return {
            "probs_key": self.probs_key,
            "save_dir": self.save_dir,
            "epoch": self.epoch,
        }

    def set_extra_state(self, state: Any) -> None:
        self.probs_key = state["probs_key"]
        self.save_dir = state["save_dir"]
        self.epoch = state["epoch"]

    def forward(self, tensordict: TensorDict, dataloader_idx=0, **kwargs):
        if self.skip_dataloader(dataloader_idx) or self.is_sanity_check:
            return
        # We have the additional time dimension (also for now only one step)
        batched_probs: List[List[torch.Tensor]] = tensordict[self.probs_key]
        batched_probs: List[torch.Tensor] = [ls[0].detach() for ls in batched_probs]
        state_indices = tensordict["idx_in_space"].squeeze()
        self.state_id_in_epoch[dataloader_idx].append(state_indices)
        self.probs_in_epoch[dataloader_idx].append(batched_probs)

    def on_validation_epoch_start(self, trainer, module) -> None:
        self.state_id_in_epoch.clear()
        self.probs_in_epoch.clear()

    def on_validation_epoch_end(self, trainer, module) -> None:
        for dataloader_idx, state_id_in_epoch in self.state_id_in_epoch.items():
            probs_in_epoch = self.probs_in_epoch[dataloader_idx]
            flattened_indices: torch.Tensor = torch.cat(state_id_in_epoch)
            sorted_ids, new_indices = torch.sort(flattened_indices)
            new_indices_list: List[int] = new_indices.tolist()
            flattened_probs: List[torch.Tensor] = list(
                itertools.chain.from_iterable(probs_in_epoch)
            )
            sorted_probs: List[torch.Tensor] = []
            for i in new_indices_list:
                sorted_probs.append(flattened_probs[i])

            dataloader_name = self.dataloader_names.get(dataloader_idx) or str(
                dataloader_idx
            )
            file_name = (
                self.save_dir / f"{self.log_name}_{dataloader_name}_{self.epoch}.pt"
            )
            torch.save(sorted_probs, file_name)
            self.epoch += 1
