import itertools
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

from rgnet.utils.batching import batched_permutations
from xmimir import XDomain, XPredicate
from xmimir.wrappers import XCategory, atom_str_template


def to_states_batch(
    data_list: list[Data | HeteroData],
    target_attr: str | None = None,
    exclude_keys: Iterable[str] | None = None,
    **kwargs,
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
        **kwargs: Additional keyword arguments for Batch.from_data_list.

    Returns:
        batch: A Batch object containing the current states(-data) as a pyg.Batch object.
    """
    batch = Batch.from_data_list(
        data_list, exclude_keys=exclude_keys or ["targets"], **kwargs
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
    **kwargs,
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


class StatefulCollater:
    def __init__(self, fn: Callable, **kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def __call__(self, data_list: list[Data | HeteroData], **kwargs):
        return self.fn(data_list, **self.kwargs, **kwargs)


__all__ = [
    "to_atom_values_batch",
    "to_states_batch",
    "to_transitions_batch",
    "StatefulCollater",
]
