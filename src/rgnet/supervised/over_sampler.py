from typing import List, Optional, Union

import torch.utils.data
from torch import Tensor
from torch_geometric.data import Data, Dataset, InMemoryDataset


class OverSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(
        self,
        dataset: Union[Dataset, Data, List[Data], Tensor],
        oversampled_class: int,
        oversampling_factor: float,
        num_samples: Optional[int] = None,
    ):
        # Analogous to ImbalancedSampler, but with oversampling for a specific class.

        if isinstance(dataset, Data):
            y = dataset.y.view(-1)
            assert dataset.num_nodes == y.numel()

        elif isinstance(dataset, Tensor):
            y = dataset.view(-1)

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.y.view(-1)
            assert len(dataset) == y.numel()

        else:
            ys = [data.y for data in dataset]
            if isinstance(ys[0], Tensor):
                y = torch.cat(ys, dim=0).view(-1)
            else:
                y = torch.tensor(ys).view(-1)
            assert len(dataset) == y.numel()

        assert y.dtype == torch.long

        num_samples = y.numel() if num_samples is None else num_samples

        # weight of sample = inverse of frequency of label
        class_weight = 1.0 / y.bincount()
        weight = class_weight[y]

        # Boost the oversampled class with higher weight
        oversampled_indices = y == oversampled_class
        weight[oversampled_indices] *= oversampling_factor

        super().__init__(weight, num_samples, replacement=True)
