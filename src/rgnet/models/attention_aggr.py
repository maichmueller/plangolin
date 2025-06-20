import math
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import Aggregation
from torch_geometric.utils import softmax


class AttentionAggregation(Aggregation):
    """
    Attention-based global pooling for graph-level readout.

    Aggregates node features x (shape [N, F]) into graph representations (shape [B, F_out])
    by computing attention scores per node and performing a weighted sum over nodes in each graph.

    Follows a single- or multi-head attention mechanism. Let F be the feature size:
        scores_h = softmax(Q_h @ K / sqrt(d_k)) * V  # shape [F] for each head h <= H
        output = W_out @ (scores_0 * values, scores_1 * values, ..., scores_H * values)  # shape [F]

    where:
        - Q is the query matrix, each row the query of that head  # shape [H, F] learnable parameters
        - K is the key vector of each node
        - V is the value vector of each node
        - d_k is the dimension of the key vectors (usually F/2 if split)
        - W_out is a linear projection to output feature size

    Args:
        feature_size (int): Dimensionality of node features.
        num_heads (int, optional): Number of attention heads. Default: 1.
        split_features (bool, optional): If True, splits features into keys and values halves.
                                         Otherwise, uses full feature for both. Default: True.
    """

    def __init__(
        self,
        feature_size: int,
        num_heads: int = 1,
        split_features: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        assert self.num_heads >= 1
        self.split_features = split_features

        if split_features:
            assert feature_size % 2 == 0
            self.key_dim = self.value_dim = feature_size // 2
        else:
            self.key_dim = self.value_dim = feature_size

        self.queries = torch.nn.Linear(self.key_dim, num_heads, bias=False)
        self.scale = 1.0 / math.sqrt(self.key_dim)

        if num_heads > 1:
            # to re-project concatenated heads back to feature_size
            self.project = torch.nn.Linear(num_heads * self.value_dim, feature_size)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> torch.Tensor:
        # split into keys & values
        if self.split_features:
            keys, values = x[:, : self.key_dim], x[:, self.key_dim :]
        else:
            keys = values = x

        # compute & normalize attention scores
        scores = self.queries(keys) * self.scale
        attn = softmax(scores, index, ptr, dim_size, dim)

        # weight values
        if self.num_heads > 1:
            attn = attn.unsqueeze(-1)  # [N, H, 1]
            vals = values.unsqueeze(1).expand(-1, self.num_heads, self.value_dim)
            weighted = attn * vals  # [N, H, D]
            weighted = weighted.view(x.size(0), -1)  # [N, H*D]
            out = self.project(weighted)  # [N, F]
        else:
            out = attn * values  # [N, D]

        # sum per graph
        return self.reduce(out, index, ptr, dim_size, dim, reduce="sum")


class AttentionPooling(torch.nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_heads: int = 1,
        split_features: bool = True,
    ) -> None:
        super().__init__()
        self.aggregator = AttentionAggregation(feature_size, num_heads, split_features)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor, size: int | None = None
    ) -> torch.Tensor:
        # batch: [N] tensor mapping each node to its graph in the batch
        return self.aggregator(x, index=batch, dim_size=size)
