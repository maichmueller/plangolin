import itertools
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

from xmimir import XPredicate
from xmimir.wrappers import atom_str_template


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
    predicates: Sequence[XPredicate],
    unreachable_atom_value: float,
    **kwargs,
) -> tuple[Batch, dict[str, Tensor], dict[str, list[tuple[int, str]]]]:
    """
    Collate function for batching atom distances in PyTorch Geometric.

    Args:
        data_list (list): List of data objects to be batched.
        **kwargs: Additional keyword arguments.

    Returns:
        batch: A Batch object containing the atom distances as a pyg.Batch object.
    """
    states_batch = to_states_batch(data_list, exclude_keys=["targets", "atom_values"])
    target_atom_values = dict()
    target_state_association = dict()
    for arity in sorted(set(p.arity for p in predicates)):
        atom_objects_of_arity = []
        state_association = []
        for data_idx, data_entry in enumerate(data_list):
            object_names = data_entry.object_names
            for object_permutation_batch in batched_permutations(
                object_names,
                arity,
                batch_size=1,
            ):
                state_association.extend([data_idx] * len(object_permutation_batch))
                for (
                    permutation_indices,
                    object_permutation,
                    _,
                ) in object_permutation_batch:
                    atom_objects_of_arity.append(
                        tuple(object_names[index] for index in permutation_indices)
                    )
        for predicate in sorted(p.name for p in predicates if p.arity == arity):
            target_values = []
            association = []
            for data in data_list:
                atom_values = data.atom_values
                for state_index, objects in zip(
                    state_association, atom_objects_of_arity
                ):
                    atom = atom_str_template.render(
                        predicate=predicate, objects=objects
                    )
                    value = atom_values[atom]
                    target_values.append(
                        value
                        if value not in (float("inf"), float("-inf"))
                        else unreachable_atom_value
                    )
                    association.append((state_index, atom))
            target_atom_values[predicate] = torch.tensor(
                target_values, dtype=torch.float
            )
            target_state_association[predicate] = association
    return states_batch, target_atom_values, target_state_association


__all__ = [
    "to_atom_values_batch",
    "to_states_batch",
    "to_transitions_batch",
]
