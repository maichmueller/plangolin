import math
from typing import Optional

import torch
import torch_geometric
from torch import Tensor
from torch_geometric.utils import softmax


class AttentionAggregation(torch_geometric.nn.Aggregation):
    """
    Attention pooling layer that aggregates node features using attention scores.

    Follows the standard attention mechanism where each node's feature is weighted
    by a learned attention score, allowing the model to focus on the most relevant
    nodes in the graph. The formula is:

    .. math::
        \text{output} = \sum_{i} softmax(Q * K^T / \sqrt(d)) \cdot x_i

    for a single head attention, where :math:`Q` is the query vector of this head and :math:`x_i`
    is the feature vector of node :math:`i`. For multiple heads, the output is concatenated across
    heads (:math:`Q` is a matrix of queries in this case) and then projected back to original feature size.

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
            assert (
                feature_size % 2 == 0
            ), "Feature size must be divisible in two halves."
            self.feature_size = feature_size // 2
        else:
            self.feature_size = feature_size
        self.queries = torch.nn.Linear(self.feature_size, num_heads, bias=False)
        self.dim_normalize_constant = 1.0 / math.sqrt(self.feature_size)

        if self.num_heads > 1:
            self.project = torch.nn.Linear(num_heads * self.feature_size, feature_size)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        if self.split_features:
            keys = x[..., : self.feature_size]
            values = x[..., self.feature_size :]
        else:
            keys = values = x
        scores = self.queries(keys) * self.dim_normalize_constant
        scores = softmax(scores, index, ptr, dim_size, dim)
        if self.num_heads > 1:
            scores = scores.view(-1, self.num_heads, 1)
            values = values.unsqueeze(1).expand(-1, self.num_heads, self.feature_size)
            # this is why we need to have evenly splittable feature sizes,
            # the elementwise multiplication would not work otherwise
            att_values = scores * values
            att_values = self.project(
                att_values.view(-1, self.num_heads * self.feature_size)
            )
        else:
            att_values = scores * values
        return self.reduce(att_values, index, ptr, dim_size, dim, reduce="sum")
