from __future__ import annotations

from typing import Callable, Union

import torch
import torch_geometric as pyg
from tensordict import TensorDict
from torch import Tensor


class ObjectEmbedding:

    dense_embedding: Tensor
    is_real_mask: Tensor

    def __init__(self, dense_embedding: Tensor, is_real_mask: Tensor) -> None:
        self.dense_embedding = dense_embedding
        self.is_real_mask = is_real_mask

    def to_tensordict(self) -> TensorDict:
        return TensorDict(
            {
                "dense_embedding": self.dense_embedding,
                "is_real_mask": self.is_real_mask,
            },
            batch_size=(self.dense_embedding.size(0),),
        )

    @staticmethod
    def embeddings_is_close(
        embeddings: TensorDict | ObjectEmbedding, other: TensorDict | ObjectEmbedding
    ) -> bool:
        if isinstance(embeddings, TensorDict):
            embeddings = ObjectEmbedding.from_tensordict(embeddings)
        if isinstance(other, TensorDict):
            other = ObjectEmbedding.from_tensordict(other)
        return (
            torch.allclose(embeddings.dense_embedding, other.dense_embedding)
            and (embeddings.is_real_mask == other.is_real_mask).all()
        )

    @staticmethod
    def from_tensordict(tensordict: TensorDict) -> ObjectEmbedding:
        return ObjectEmbedding(
            tensordict["dense_embedding"], tensordict["is_real_mask"]
        )

    @staticmethod
    def from_sparse(sparse_embedding: torch.Tensor, batch: torch.Tensor):
        dense, mask = pyg.utils.to_dense_batch(
            sparse_embedding, batch, fill_value=torch.nan
        )
        return ObjectEmbedding(dense_embedding=dense, is_real_mask=mask)


def mask_to_batch_indices(mask: torch.Tensor):
    """
    Given the mask indicating which objects are real, return the batch indices.
    This can be seen as the inverse of `torch_geometric.utils.to_dense_batch`.
    batch[i] = j means that the i-th object belongs to the j-th graph.
    :param mask: A boolean mask indicating which objects are real.
        Can be of with or without batch dimension.
    :return: A tensor of batch indices. shape [N] where N is the number of objects.
    """
    return torch.arange(mask.size(0), device=mask.device).repeat_interleave(
        mask.sum(dim=-1)
    )


class ObjectPoolingModule(torch.nn.Module):

    @staticmethod
    def nansum(dense_embedding: torch.Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def nanmean(dense_embedding: torch.Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def nanmax(dense_embedding: torch.Tensor):
        return dense_embedding.nan_to_num(torch.finfo(dense_embedding.dtype).min).max(
            dim=-2
        )

    def __init__(
        self, pooling: Union[str, Callable[[Tensor, Tensor], Tensor]] = "add"
    ) -> None:
        super().__init__()
        self.is_pooling_str: bool = isinstance(pooling, str)
        if self.is_pooling_str:
            if pooling == "add" or pooling == "sum":
                self.pooling = ObjectPoolingModule.nansum
            elif pooling == "mean":
                self.pooling = ObjectPoolingModule.nanmean
            elif pooling == "max":
                self.pooling = ObjectPoolingModule.nanmax
            else:
                raise ValueError(
                    f"Unknown pooling function: {pooling}. Choose from [add, mean, max]."
                )
        else:
            self.pooling = pooling

    def forward(self, object_embedding: ObjectEmbedding | TensorDict | torch.Tensor):
        if isinstance(object_embedding, TensorDict):
            object_embedding = ObjectEmbedding.from_tensordict(object_embedding)
        if isinstance(object_embedding, torch.Tensor):
            assert self.is_pooling_str
            dense_embedding = object_embedding
        else:
            dense_embedding = object_embedding.dense_embedding

        assert dense_embedding.dim() >= 2

        if self.is_pooling_str:
            return self.pooling(dense_embedding)

        return self.pooling(
            dense_embedding[object_embedding.is_real_mask],
            mask_to_batch_indices(object_embedding.is_real_mask),
        )
