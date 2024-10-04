import itertools

import torch
from torch_geometric.data import Batch


def collate_fn(data_list, **kwargs):
    flattened_targets = list(
        itertools.chain.from_iterable([d.targets for d in data_list])
    )

    successor_batch = Batch.from_data_list(flattened_targets)

    batched = Batch.from_data_list(data_list, exclude_keys=["targets"])

    num_successors = torch.tensor(
        [data.reward.numel() for data in data_list], dtype=torch.long
    )

    return batched, successor_batch, num_successors
