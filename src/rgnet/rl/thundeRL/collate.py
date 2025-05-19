import itertools
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

from rgnet.utils.batching import batched_permutations
from xmimir import XDomain, XPredicate
from xmimir.wrappers import XCategory, atom_str_template


def to_states_batch(
    data_list: list[Data | HeteroData],
    exclude_keys: Iterable[str] | None = None,
    **kwargs,
) -> Batch:
    """
    Collate function for batching current states-(data) in PyTorch Geometric.

    Args:
        data_list (list): List of data objects to be batched.
        exclude_keys (tuple): Keys to exclude from the batch.
        **kwargs: Additional keyword arguments.

    Returns:
        batch: A Batch object containing the current states(-data) as a pyg.Batch object.
    """
    return Batch.from_data_list(
        data_list, exclude_keys=exclude_keys or ["targets"], **kwargs
    )


def to_transitions_batch(
    data_list: list[Data | HeteroData], **kwargs
) -> tuple[Batch, Batch, Tensor]:
    batched = to_states_batch(data_list, **kwargs)

    flattened_targets = list(
        itertools.chain.from_iterable(d.targets for d in data_list)
    )
    num_successors = torch.tensor(
        [len(data.targets) for data in data_list], dtype=torch.long
    )
    successor_batch = to_states_batch(flattened_targets, **kwargs)

    return batched, successor_batch, num_successors


def to_atom_values_batch(
    data_list: list[Data | HeteroData],
    predicates: Sequence[XPredicate] | XDomain,
    unreachable_atom_value: float,
    **kwargs,
) -> tuple[Batch, dict[str, Tensor], dict[str, list[tuple[int, str]]]]:
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
    states_batch = to_states_batch(data_list, exclude_keys=["targets", "atom_values"])
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
                atom_values = data.atom_values
                for objects in atom_objects_permutations:
                    atom = atom_str_template.render(
                        predicate=predicate, objects=objects
                    )
                    value = atom_values[atom]
                    target_values.append(
                        value
                        if value not in (float("inf"), float("-inf"))
                        else unreachable_atom_value
                    )
                    association.append((data_idx, atom))
            target_atom_values[predicate] = torch.tensor(
                target_values, dtype=torch.float
            )
            target_state_association[predicate] = association
    return states_batch, target_atom_values, target_state_association


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
