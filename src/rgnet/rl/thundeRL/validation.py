import abc
import logging
import statistics
import warnings
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set

import torch
import torch_geometric as pyg
from lightning import Callback
from tensordict import NestedKey, TensorDict
from torch import Tensor
from torchrl.modules import ValueOperator
from tqdm import tqdm

from rgnet.rl.agents import ActorCritic
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.envs.planning_env import PlanningEnvironment
from rgnet.rl.policy_evaluation import (
    PolicyEvaluationMessagePassing,
    build_mdp_graph,
    mdp_graph_as_pyg_data,
)
from rgnet.utils.utils import KeyAwareDefaultDict


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
        :param dataloader_names: Mapping each data loader to a string.
            The names will be used in the full logg-key.
            The full key will be val/{log_name}_{dataloader_names[idx].
             If not specified, the index of the dataloader will be used instead.
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
        return f"val/{self.log_name}_" + (
            self.dataloader_names[dataloader_idx]
            if self.dataloader_names
            else str(dataloader_idx)
        )

    def skip_dataloader(self, dataloader_index):
        return (
            self.only_run_for_dataloader is not None
            and dataloader_index not in self.only_run_for_dataloader
        )

    @abc.abstractmethod
    def forward(
        self, tensordict: TensorDict, batch_idx: Optional[int] = None, dataloader_idx=0
    ):
        pass


class CriticValidation(ValidationCallback):
    def __init__(
        self,
        discounted_optimal_values: Dict[int, torch.Tensor],
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
        :param discounted_optimal_values: Dictionary mapping the dataloader index to a flat tensor.
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
        for space_idx, optimal_values in discounted_optimal_values.items():
            self.register_buffer(str(space_idx), optimal_values)
        self.value_op = value_operator
        self.loss_function = loss_function or torch.nn.functional.mse_loss
        self.state_value_key = state_value_key

    def forward(self, tensordict: TensorDict, batch_idx=None, dataloader_idx=0):
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

    def forward(self, tensordict: TensorDict, batch_idx=None, dataloader_idx=0):
        """
        Assesses the policy as the number of states in which an optimal action was chosen
        divided by the number of states.
        Two alternatives could be considered too:
         - Factor in the number of transitions as denominator
         - Use the actual probabilities and not just the sampled action
        :param tensordict: Result of agent on environment. Specifically, we expect
            - "idx_in_space" identifying the current state in the respective StateSpace
            - PlanningEnvironment.keys.action the idx of the transition which the agent chose
        :param batch_idx: Not required for this callback
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
        :return: Tensor of shape [1, ] on the same device as the input tensor.
        """
        if probs_tensor.numel() == 1:
            return torch.zeros((1,), dtype=torch.float, device=probs_tensor.device)
        entropy = torch.distributions.Categorical(probs=probs_tensor).entropy()
        return entropy / torch.tensor(
            (probs_tensor.numel(),), device=probs_tensor.device
        )

    def forward(self, tensordict: TensorDict, batch_idx=None, dataloader_idx=0):
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


class ProbsCollector(torch.nn.Module):
    """
    Used to collect all probabilities over one epoch.
    The probabilities are then sorted and provided as a continuous list, where
    self.sorted_epoch_probs[i][j][k] is
        the state-space corresponding to the i-th dataloader,
        the j-th state in the state-space,
        the k-th successor of this state.

    Multiple validation-callbacks can share a common collector.
    For this use case, the batch_idx has to be passed, which will be used to identify multiple calls with the same data.
    The collector will hold the probability tensor on the same device, as received in forward.
    """

    def __init__(
        self,
        probs_key: NestedKey = ActorCritic.default_keys.probs,
    ):
        super().__init__()
        self.probs_key: NestedKey = probs_key
        # Collected over one validation epoch
        # The index of the current states in their StateSpaces collected over one epoch
        self._state_id_in_epoch: Dict[int, List[torch.Tensor]] = defaultdict(list)
        # The probability for each outgoing transition over one epoch
        # Each state can have various numbers of successors, therefore, we have a list of list
        self._probs_in_epoch: Dict[int, List[List[torch.Tensor]]] = defaultdict(list)

        self._sorted_probs: Optional[Dict[int, List[torch.Tensor]]] = None
        self._seen_batch_indices: Dict[int, Set[int]] = defaultdict(set)

    def forward(self, tensordict: TensorDict, batch_idx: int, dataloader_idx=0):
        """
        Save the transition probabilities over one epoch. Also works with shuffled data.
        The `idx_in_space` entry will be used to order the collected probabilities.
        :param tensordict: Of shape [batch_size, 1, feature_dim] with to keys
            "idx_in_space" and `self.probs_key` present.
            The "idx_in_space" has to contain the index of each present state, over one
            full epoch every index should occur only once and be from 0,...,N-1 for some N.
            The probs entry should contain the probability distribution for each outgoing
             transition for each state in the batch.
        :param batch_idx: Used to detect multiple calls to forward with the same data.
            Every following call with the same batch_idx will be ignored until `reset` is called.
        :param dataloader_idx: Optional parameter if multiple state spaces are used for validation.
        """
        if batch_idx in self._seen_batch_indices[dataloader_idx]:
            return
        self._seen_batch_indices[dataloader_idx].add(batch_idx)
        if self._sorted_probs is not None:
            raise UserWarning(
                "You hae to reset the collector after each call to "
                "sort_probs_on_epoch_end() before adding new data."
            )
        # We have the additional time dimension (also for now only one step)
        batched_probs: List[List[torch.Tensor]] = tensordict[self.probs_key]
        batched_probs: List[torch.Tensor] = [ls[0].detach() for ls in batched_probs]
        state_indices = tensordict["idx_in_space"].squeeze()

        # Validate inputs
        if len(batched_probs) != state_indices.numel():
            raise ValueError(
                "Number of probability tensors must match number of states"
            )

        self._state_id_in_epoch[dataloader_idx].append(state_indices)
        self._probs_in_epoch[dataloader_idx].append(batched_probs)

    def reset(self) -> None:
        """The collector should be reset at the start of every epoch."""
        self._state_id_in_epoch.clear()
        self._probs_in_epoch.clear()
        self._seen_batch_indices.clear()
        self._sorted_probs = None

    def sort_probs(self, dataloader_idx: int):
        probs_in_epoch: List[List[Tensor]] = self._probs_in_epoch[dataloader_idx]
        flattened_indices: torch.Tensor = torch.cat(
            self._state_id_in_epoch[dataloader_idx]
        )
        sorted_ids, new_indices = torch.sort(flattened_indices)
        flattened_probs: List[torch.Tensor] = list(chain.from_iterable(probs_in_epoch))
        sorted_probs: List[torch.Tensor] = [
            flattened_probs[i] for i in new_indices.tolist()
        ]
        if self._sorted_probs is None:
            self._sorted_probs = {dataloader_idx: sorted_probs}
        return sorted_probs

    def sort_probs_on_epoch_end(self) -> Dict[int, List[Tensor]]:
        """
        This method should be called inside on_validation_epoch_end.
        The method will sort the collected probabilities such that the first tensor
        contains the probabilities for the successors of the first state.
        NOTE if the collector is shared, the dictionary will contain values for all
        dataloader indices. You might want to run skip_dataloader again
         while iterating over the values.

        """
        if self._sorted_probs:
            return self._sorted_probs

        self._sorted_probs = {
            dataloader_idx: self.sort_probs(dataloader_idx)
            for dataloader_idx in self._state_id_in_epoch.keys()
        }
        return self._sorted_probs


class ProbsStoreCallback(ValidationCallback):
    def __init__(
        self,
        save_dir: Path,
        probs_collector: ProbsCollector,
        log_name: str = "actor_probs",
        dataloader_names: Optional[Dict[int, str]] = None,
        only_run_for_dataloader: Optional[set[int]] = None,
    ):
        """
        Save the probability for all successors for every state.
        The combined probabilities over one epoch are saved to a .pt file.
        The saved file is of the form List[torch.Tensor]. save_file[i][j] is the probability
        in statespace.get_states()[i] for moving to the j-th successor.

        :param probs_collector: The collector which aggregates probs over one epoch.
        :param save_dir: Where to store the .pt files.
        :param log_name: Start of the file-names. (default: actor_probs)
        """
        super().__init__(
            log_name=log_name,
            dataloader_names=dataloader_names,
            only_run_for_dataloader=only_run_for_dataloader,
        )
        self.probs_collector: ProbsCollector = probs_collector
        assert isinstance(save_dir, Path)
        self.save_dir: Path = save_dir / self.log_name
        self.save_dir.mkdir(exist_ok=True)
        self.epoch: int = 0

    def get_extra_state(self) -> Any:
        return {
            "save_dir": self.save_dir,
            "epoch": self.epoch,
        }

    def set_extra_state(self, state: Any) -> None:
        self.save_dir = state["save_dir"]
        self.epoch = state["epoch"]

    def forward(
        self, tensordict: TensorDict, batch_idx: Optional[int] = 0, dataloader_idx=0
    ):
        if self.skip_dataloader(dataloader_idx):
            return
        self.probs_collector(tensordict, batch_idx, dataloader_idx)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.probs_collector.reset()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for (
            dataloader_idx,
            epoch_probs,
        ) in self.probs_collector.sort_probs_on_epoch_end().items():
            if self.skip_dataloader(dataloader_idx):  # collector can be shared
                continue
            dataloader_name = self.dataloader_names.get(dataloader_idx) or str(
                dataloader_idx
            )

            file_name = (
                self.save_dir / f"{self.log_name}_{dataloader_name}_{self.epoch}.pt"
            )
            torch.save(epoch_probs, file_name)
            self.epoch += 1


class PolicyEvaluationValidation(ValidationCallback):
    """
     Evaluate the current policy by determining the value function induced by the policy
     and comparing it to the optimal value function.
     The policy evaluation algorithm will compute the discounted value for each state.
     During the epoch the transition-probabilities for each state in each validation space
        are collected using the probs_collector.
    At the end of each validation epoch, the policy evaluation is executed.
    The evaluation will either run for num_iterations or until the change is smaller than the threshold.
    Adding the callback as a validation_hook to the PolicyGradientModule ensures policy evaluation runs on the same device as model training.
    """

    def __init__(
        self,
        envs: List[ExpandedStateSpaceEnv],
        discounted_optimal_values: Dict[int, torch.Tensor],
        probs_collector: ProbsCollector,
        log_name: str = "policy_evaluation",
        num_iterations: Optional[int] = 1_000,
        difference_threshold: Optional[float] = 0.001,
        log_aggregated_metric: bool = True,
        dataloader_names: Optional[Dict[int, str]] = None,
        only_run_for_dataloader: Optional[set[int]] = None,
    ) -> None:
        """
        :param spaces: List of all validation spaces (independent of only_run_for_dataloader)
            This is needed to construct the state space as networkx graph.
        :param discounted_optimal_values: Dictionary mapping the dataloader index to a flat tensor.
            optimal_values_dict[i][j] is the optimal value for the j-th state in the i-th validation space.
        :param probs_collector: Potentially shared collector used to gather the probabilities.
        :param log_name: Name under which the metric results will be logged.
        :param num_iterations: An optional upper limit for the policy evaluation.
            (default 1_000)
        :param difference_threshold: Optional L1-norm difference threshold for early stopping.
            Policy evaluation halts if the change in the value function is less than this threshold.
            (default 0.001)
        :param log_aggregated_metric: Whether an additional metric val/<log_name> should be logged that averages
            the result of all used validation problems. Simplifies `ModelCheckpoint` usage by simply specifying `monitor: val/policy_evaluation`.
        Both num_iterations and difference_threshold can be used simultaneously, but at least one
        has to be specified.
        """
        super().__init__(
            log_name, dataloader_names, only_run_for_dataloader, epoch_reduction="mean"
        )
        if num_iterations is None and difference_threshold is None:
            raise ValueError(
                "Neither num_iterations nor difference_threshold was given."
                "At least one is required to determine value-iteration limit."
            )
        gamma = envs[0].reward_function.gamma
        assert all(
            env.reward_function.gamma == gamma
            for i, env in enumerate(envs)
            if not self.skip_dataloader(i)
        ), "Gamma has to be the same for all validation spaces."

        self.log_aggregated_metric = log_aggregated_metric
        self.probs_collector = probs_collector
        for space_idx, optimal_values in discounted_optimal_values.items():
            self.register_buffer(str(space_idx), optimal_values)

        self._last_seen_dataloader_idx: int = 0
        self._losses = KeyAwareDefaultDict(
            lambda dataloader_idx: self._compute_loss(dataloader_idx)
        )
        self._graphs: Dict[int, pyg.data.Data] = dict()
        logging.info("Building state space graphs for validation problem.")
        for idx, env in tqdm(enumerate(envs), total=len(envs)):
            if self.skip_dataloader(idx):
                continue
            nx_graph = build_mdp_graph(env)
            self._graphs[idx] = mdp_graph_as_pyg_data(nx_graph)

        self.message_passing = PolicyEvaluationMessagePassing(
            gamma=gamma,
            num_iterations=num_iterations,
            difference_threshold=difference_threshold,
        )

    def _apply(self, fn, recurse=True):
        """
        We need to transfer the graphs to device which are non-tensors, so we can't use register_buffer().
        The functions to(...) or cuda(...) are actually never called because the callbacks
        are only nested within policy_gradient_lit_module so this seems to be the only way to
        catch the device transfer.
        """
        if recurse:
            for graph in self._graphs.values():
                graph.apply(fn)
        return super()._apply(fn, recurse)

    def compute_values(
        self, probs_list: List[torch.Tensor], dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute the policy evaluation by setting the correct probs and calling the
        internal message passing module.

        :param probs_list: The sorted transition probabilities for each state.
            Len(probs_list) = space.get_states() for state space of dataloader_idx
            probs_list[i].numel() == len(space.iter_forward_transitions(space.get_states()[i]))
        :param dataloader_idx: Required if the validator is used with multiple spaces at the same time.
        :return: A one-dimensional tensor with the values under the provided probabilities.
        """
        # The first entry of edge attributes are the transition probabilities.
        flat_probs = torch.cat(probs_list)
        graph: pyg.data.Data = self._graphs[dataloader_idx]
        assert (
            flat_probs.shape == graph.edge_attr[:, 0].shape
        ), f"Found mismatching shapes {flat_probs.shape=} and {graph.edge_attr[:, 0].shape=}"
        assert (
            flat_probs.device == graph.edge_attr.device
        ), f"Found mismatching devices {flat_probs.device=} and {graph.edge_attr.device=}"
        graph.edge_attr[:, 0] = flat_probs
        return self.message_passing(graph)

    def forward(
        self,
        tensordict: TensorDict,
        batch_idx: Optional[int] = 0,
        dataloader_idx: int = 0,
    ):
        if self.skip_dataloader(dataloader_idx) or self.is_sanity_check:
            return

        prev_dataloader_idx = self._last_seen_dataloader_idx
        if dataloader_idx != prev_dataloader_idx:
            assert dataloader_idx > prev_dataloader_idx
            loss = self._compute_loss(prev_dataloader_idx).cpu().detach()
            self._losses[prev_dataloader_idx] = loss
            self._last_seen_dataloader_idx = dataloader_idx
            self.probs_collector.reset()

        self.probs_collector(
            tensordict, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )

    def _compute_loss(self, dataloader_idx) -> Optional[torch.Tensor]:
        epoch_probs = self.probs_collector.sort_probs(dataloader_idx)
        values = self.compute_values(epoch_probs, dataloader_idx)
        if self.skip_dataloader(dataloader_idx):  # collector can be shared
            return None
        try:
            optimal_values: torch.Tensor = self.get_buffer(str(dataloader_idx))
        except AttributeError:
            warnings.warn(
                f"No optimal values found for dataloader_idx {dataloader_idx}"
            )
            return None
        assert values.shape == optimal_values.shape
        assert values.device == optimal_values.device
        loss = torch.nn.functional.l1_loss(values, optimal_values)
        return loss

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.probs_collector.reset()
        self._losses.clear()
        self._last_seen_dataloader_idx = 0

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        losses: List[torch.Tensor] = []
        sorted_probs = self.probs_collector.sort_probs_on_epoch_end()
        for dataloader_idx in filter(
            lambda i: not self.skip_dataloader(i), sorted_probs.keys()
        ):
            loss = self._losses[dataloader_idx]
            pl_module.log(
                self.log_key(dataloader_idx),
                loss,
                on_epoch=True,
            )
            losses.append(loss)

        if self.log_aggregated_metric:
            pl_module.log(
                f"val/{self.log_name}",
                torch.stack(losses).view(-1).mean(),
                on_epoch=True,
            )
