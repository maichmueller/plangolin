import itertools
from typing import Iterable

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData


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
    data_list: list[Data | HeteroData], **kwargs
) -> tuple[Batch, Tensor]:
    """
    Collate function for batching atom distances in PyTorch Geometric.

    Args:
        data_list (list): List of data objects to be batched.
        **kwargs: Additional keyword arguments.

    Returns:
        batch: A Batch object containing the atom distances as a pyg.Batch object.
    """
    states_batch = to_states_batch(data_list, exclude_keys=["targets", "atom_values"])
    atom_values = torch.tensor([data.atom_values for data in data_list])
    return states_batch, atom_values


__all__ = [
    "to_atom_values_batch",
    "to_states_batch",
    "to_transitions_batch",
]
