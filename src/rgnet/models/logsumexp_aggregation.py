from typing import Optional

import torch
import torch_geometric.nn
from torch import Tensor


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
