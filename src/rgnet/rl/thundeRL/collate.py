import inspect
import itertools
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

import xmimir as xmi
from rgnet.encoding import EncoderFactory
from rgnet.rl.envs import SuccessorEnvironment
from rgnet.rl.reward import RewardFunction
from rgnet.utils.batching import batched_permutations
from rgnet.utils.misc import KeyAwareDefaultDict
from xmimir import XDomain, XPredicate, XSuccessorGenerator, XTransition, iw
from xmimir.wrappers import XCategory, atom_str_template


def to_states_batch(
    data_list: list[Data | HeteroData],
    target_attr: str | None = None,
    exclude_keys: Optional[List[str]] = None,
    follow_batch: Optional[List[str]] = None,
    **ignore,
) -> (
    tuple[Batch, dict[str, Any]]
    | tuple[Batch, Tensor | Sequence | Mapping, dict[str, Any]]
):
    """
    Collate function for batching current states-(data) in PyTorch Geometric.

    Args:
        data_list (list): List of data objects to be batched.
        target_attr (str | None): The attribute to use as targets, if any.
        exclude_keys (tuple): Keys to exclude from the batch.
        follow_batch (list): List of keys to follow for batching.

    Returns:
        batch: A Batch object containing the current states(-data) as a pyg.Batch object.
    """
    batch = Batch.from_data_list(
        data_list,
        exclude_keys=exclude_keys or ["targets"],
        follow_batch=follow_batch,
    )
    info = {
        "batch_size": batch.batch_size,
    }
    if target_attr is None:
        return batch, info
    else:
        targets = getattr(batch, target_attr)
        return batch, targets, info


def to_transitions_batch(
    data_list: list[Data | HeteroData], **kwargs
) -> tuple[Batch, Batch, Tensor, dict[str, Any]]:
    batched, info = to_states_batch(data_list, **kwargs)

    flattened_targets = list(
        itertools.chain.from_iterable(d.targets for d in data_list)
    )
    num_successors = torch.tensor(
        [len(data.targets) for data in data_list], dtype=torch.long
    )
    successor_batch, succ_info = to_states_batch(flattened_targets, **kwargs)

    return (
        batched,
        successor_batch,
        num_successors,
        info | {f"successors_{key}": value for key, value in succ_info.items()},
    )


def to_atom_values_batch(
    data_list: list[Data | HeteroData],
    predicates: Sequence[XPredicate] | XDomain,
    unreachable_atom_value: float,
    **ignore,
) -> tuple[Batch, dict[str, Tensor], dict[str, list[tuple[int, str]]], dict[str, Any]]:
    """
    Collate function for batching atom distances in PyTorch Geometric.


    :params: data_list (list): List of data objects to be batched.
    :params: predicates (Sequence[XPredicate]): List of predicates to be used for atom values.
    :params: unreachable_atom_value (float): Value to be used for unreachable atoms.

    Returns:
        batch: A Batch object containing the atom distances as a pyg.Batch object.
    """
    if isinstance(predicates, XDomain):
        predicates = predicates.predicates(XCategory.fluent, XCategory.derived)
    states_batch, info = to_states_batch(
        data_list, exclude_keys=["targets", "atom_values_dict", "atom_values_tensor"]
    )
    target_atom_values = dict()
    target_state_association = dict()
    for arity in sorted(set(p.arity for p in predicates)):
        atom_objects_of_arity = []
        if arity == 0:
            atom_objects_of_arity = [[None]] * len(data_list)
        else:
            for data_idx, data_entry in enumerate(data_list):
                object_names = data_entry.object_names
                object_permutation_batch = next(
                    batched_permutations(
                        object_names, arity, batch_size=None, with_replacement=True
                    )
                )
                atom_objects_of_arity.append(object_permutation_batch.data)
        for predicate in sorted(p.name for p in predicates if p.arity == arity):
            target_values = []
            association = []
            for data_idx, (data, atom_objects_permutations) in enumerate(
                zip(data_list, atom_objects_of_arity)
            ):
                atom_values_dict = data.atom_values_dict
                for objects in atom_objects_permutations:
                    atom = atom_str_template.render(
                        predicate=predicate, objects=objects
                    )
                    if atom not in atom_values_dict:
                        # augment with unreachable atom value if this atom is not present in this state
                        # (can be for various reasons, e.g. our object permutation augmentation considers
                        # atoms that are not found during data collection)
                        value = unreachable_atom_value
                    else:
                        value = atom_values_dict[atom]
                        if value == float("inf") or value == float("-inf"):
                            value = unreachable_atom_value
                    target_values.append(value)
                    association.append((data_idx, atom))
            target_atom_values[predicate] = torch.tensor(
                target_values, dtype=torch.float
            )
            target_state_association[predicate] = association
    return states_batch, target_atom_values, target_state_association, info


def _fetch_successor_generator(
    domain_problem_reward: tuple[str | Path, str | Path, RewardFunction],
) -> tuple[XDomain, SuccessorEnvironment]:
    """
    Fetches the successor generator and domain from the given paths.
    """
    domain_path, problem_path, reward_func = domain_problem_reward
    domain, problem = xmi.parse(domain_path, problem_path)
    return domain, SuccessorEnvironment(
        generators=[XSuccessorGenerator(problem)],
        reward_function=reward_func,
        batch_size=1,
        reset=True,
    )


def to_iw_transitions_batch(
    data_list: list[Data | HeteroData],
    iw_search: iw.IWSearch,
    reward_function: RewardFunction,
    encoder_factory: EncoderFactory,
    exclude_keys: Iterable[str] | None = None,
    **kwargs,
) -> tuple[Batch, Batch, Tensor, dict[str, Any]]:
    """
    Collate function for batching IW successors in PyTorch Geometric.

    Requires that each data object in `data_list` has an `action_history`, a "domain_path", and a "problem_path"
    attribute. These are used to reconstruct the state and to fetch the appropriate successor env, as well as
    instantiate the encoder.

    Args:
        data_list (list):
            List of data objects to be batched.
        iw_search (iw.IWSearch):
            The IW search instance to use for the worker.
        reward_function (RewardFunction):
            The reward function to use for rewarding IW successors.
        encoder_factory (EncoderFactory):
            The encoder factory to create the state-to-pyg encoder with.
        exclude_keys (tuple):
            Keys to exclude from PyG batching.
        **kwargs: Additional keyword arguments for Batch.from_data_list.

    Returns:
        batch: A Batch object containing the IW successors as a pyg.Batch object.
    """
    # Ensure the problem_dict is available in the global scope (see iw_worker_init)
    successor_gen_domain_dict: dict[
        tuple[Path, Path, RewardFunction], tuple[XDomain, SuccessorEnvironment]
    ] = KeyAwareDefaultDict(_fetch_successor_generator)
    encoders = KeyAwareDefaultDict(encoder_factory)
    for state_data in data_list:
        domain, succ_env = successor_gen_domain_dict[
            (state_data.domain_path, state_data.problem_path, reward_function)
        ]
        succ_gen = succ_env.active_instances[0]
        recon_state = state_data.action_history.reconstruct(succ_gen)
        collector = iw.CollectorHook()
        iw_search.solve(
            succ_gen,
            start_state=recon_state,
            novel_hook=collector,
            stop_on_goal=False,
        )
        encoder = encoders[domain]
        transitions, targets, rewards, done = [], [], [], []
        for node in collector.nodes:
            successor = node.state
            trace = node.trace
            iw_transition = XTransition.make_hollow(
                recon_state, [t.action for t in trace], successor
            )
            transitions.append(iw_transition)
            successor_data = encoder.to_pyg_data(encoder.encode(successor))
            targets.append(successor_data)
        reward, done = succ_env.get_reward_and_done(transitions=transitions)
        state_data.reward = reward
        state_data.done = done
        state_data.targets = targets

    return to_transitions_batch(
        data_list,
        exclude_keys=exclude_keys or ["targets"],
        **kwargs,
    )


class StatefulCollater:
    def __init__(self, fn: Callable, **kwargs):
        self.fn = fn
        signature = inspect.signature(fn)
        actual_kwargs = dict()
        for kwarg in kwargs:
            if kwarg != "self" and kwarg in signature.parameters.keys():
                param = signature.parameters[kwarg]
                if kwargs[kwarg] != param.default:
                    actual_kwargs[kwarg] = kwargs[kwarg]
        self.kwargs = actual_kwargs

    def __call__(self, data_list: list[Data | HeteroData], **kwargs):
        return self.fn(data_list, **self.kwargs, **kwargs)


__all__ = [
    "to_atom_values_batch",
    "to_states_batch",
    "to_transitions_batch",
    "to_iw_transitions_batch",
    "StatefulCollater",
]
