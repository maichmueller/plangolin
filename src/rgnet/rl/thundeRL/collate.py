import itertools

import torch
from torch_geometric.data import Batch


def collate_fn(data_list, **kwargs):
    flattened_targets = list(
        itertools.chain.from_iterable([d.targets for d in data_list])
    )

    num_successors = torch.tensor(
        [len(data.targets) for data in data_list], dtype=torch.long
    )

    successor_batch = Batch.from_data_list(flattened_targets)

    batched = Batch.from_data_list(data_list, exclude_keys=["targets"])

    return batched, successor_batch, num_successors
