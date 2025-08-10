import math
from typing import Optional

import torch
import torch_geometric.nn
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
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
        - d_k is the dimension of the key vectors (F/2 if split)
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

        if num_heads > 1 or split_features:
            # to re-project concatenated heads back to feature_size
            self.project = torch.nn.Linear(num_heads * self.value_dim, feature_size)
        else:
            self.project = torch.nn.Identity()

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
            out = weighted.view(x.size(0), -1)  # [N, H*D]
        else:
            out = attn * values  # [N, D]

        out = self.project(out)  # [N, F]

        # sum per graph
        return self.reduce(out, index, ptr, dim_size, dim, reduce="sum")


class LogSumExpAggregation(torch_geometric.nn.Aggregation):
    def __init__(self, maximum_smoothness: float = 12.0) -> None:
        super().__init__()
        self.maximum_smoothness = torch.nn.Parameter(
            torch.tensor(maximum_smoothness, dtype=torch.float), requires_grad=False
        )

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        """
        auto exps_max = torch::zeros_like(object_embeddings);
        exps_max.index_reduce_(0, output_indices, output_tensors, "amax", false);
        exps_max = exps_max.detach();

        auto exps_sum = torch::full_like(object_embeddings, 1E-16);
        const auto max_offsets = exps_max.index_select0, output_indices).detach();
        const auto exps = (maximum_smoothness_ * (output_tensors - max_offsets)).exp();
        exps_sum.index_add_(0, output_indices, exps);

        const auto max_msg = ((1.0 / maximum_smoothness_) * exps_sum.log()) + exps_max;
        const auto next_object_embeddings = object_embeddings
                                            + update_module_->forward(
                                               torch::cat({max_msg, object_embeddings}, 1)
                                            );
        """
        exps_max = torch.zeros((dim_size, x.size(-1)), device=x.device, dtype=x.dtype)
        exps_max.index_reduce_(
            dim=0, index=index, source=x, reduce="amax", include_self=False
        )
        exps_max = exps_max.detach()
        max_offsets = exps_max.index_select(0, index=index)

        exps_sum = torch.full_like(exps_max, 1e-16)
        exps = (self.maximum_smoothness * (x - max_offsets)).exp()
        exps_sum.index_add_(0, index, exps)
        max_msg = ((1.0 / self.maximum_smoothness) * exps_sum.log()) + exps_max
        return max_msg
