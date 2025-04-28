import itertools

import torch
from torch_geometric.data import Batch


def states_batching_collate_fn(data_list, **kwargs) -> Batch:
    """
    Collate function for batching current states-(data) in PyTorch Geometric.

    Args:
        data_list (list): List of data objects to be batched.
        **kwargs: Additional keyword arguments.

    Returns:
        batch: A Batch object containing the current states(-data) as a pyg.Batch object.
    """
    return Batch.from_data_list(data_list, exclude_keys=["targets"])


def transitions_batching_collate_fn(
    data_list, **kwargs
) -> tuple[Batch, Batch, torch.Tensor]:
    batched = states_batching_collate_fn(data_list, **kwargs)

    flattened_targets = list(
        itertools.chain.from_iterable(d.targets for d in data_list)
    )
    num_successors = torch.tensor(
        [len(data.targets) for data in data_list], dtype=torch.long
    )
    successor_batch = states_batching_collate_fn(flattened_targets, **kwargs)

    return batched, successor_batch, num_successors
