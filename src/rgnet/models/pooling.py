# models/pooling.py

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    MaxAggregation,
    MeanAggregation,
    Set2Set,
    SortAggregation,
    SumAggregation,
)

from rgnet.models.attention_aggr import AttentionAggregation


class GlobalPool(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: Tensor, batch: Tensor, **kwargs) -> Tensor:
        """
        Summarizes each graph to a single embedding.

        Args:
            x:     [N, F] node features
            batch: [N] graph assignment
            **kwargs: pool-specific args

        Returns:
            [num_graphs, F_out] graph-level embeddings
        """
        ...


class GlobalPoolFromAggregation(GlobalPool):
    def __init__(self, aggr: torch.nn.Module):
        super().__init__()
        self.aggr = aggr

    def forward(self, x: Tensor, batch: Tensor, **kwargs) -> Tensor:
        """
        Summarizes each graph to a single embedding using the provided aggregation method.

        Args:
            x:     [N, F] node features
            batch: [N] graph assignment
            **kwargs: pool-specific args

        Returns:
            [num_graphs, F_out] graph-level embeddings
        """
        return self.aggr(x, index=batch, **kwargs)


class GlobalMeanPool(GlobalPoolFromAggregation):
    def __init__(self):
        super().__init__(MeanAggregation())


class GlobalMaxPool(GlobalPoolFromAggregation):
    def __init__(self):
        super().__init__(MaxAggregation())


class GlobalAddPool(GlobalPoolFromAggregation):
    def __init__(self):
        super().__init__(SumAggregation())


class GlobalSortPool(GlobalPoolFromAggregation):
    def __init__(self, k: int):
        super().__init__(SortAggregation(k=k))


class GlobalAttentionPool(GlobalPoolFromAggregation):
    def __init__(
        self,
        feature_size: int,
        num_heads: int = 1,
        split_features: bool = False,
    ):
        super().__init__(AttentionAggregation(feature_size, num_heads, split_features))


class GlobalAttentionalPool(GlobalPoolFromAggregation):
    def __init__(
        self,
        gate_nn: torch.nn.Module,
        nn: torch.nn.Module | None = None,
    ):
        super().__init__(AttentionalAggregation(gate_nn=gate_nn, nn=nn))


class GlobalSet2SetPool(GlobalPoolFromAggregation):
    def __init__(self, in_channels: int, processing_steps: int, **kwargs):
        super().__init__(Set2Set(in_channels, processing_steps, **kwargs))
