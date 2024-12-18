from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch_geometric as pyg
from tensordict import TensorDict
from torch import Tensor


class ObjectEmbedding:
    dense_embedding: Tensor
    padding_mask: Tensor

    def __init__(self, dense_embedding: Tensor, padding_mask: Tensor) -> None:
        self.dense_embedding = dense_embedding
        self.padding_mask = padding_mask

    def to_tensordict(self) -> TensorDict:
        return TensorDict(
            {
                "dense_embedding": self.dense_embedding,
                "padding_mask": self.padding_mask,
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
            and (embeddings.padding_mask == other.padding_mask).all()
        )

    @staticmethod
    def from_tensordict(tensordict: TensorDict) -> ObjectEmbedding:
        return ObjectEmbedding(
            tensordict["dense_embedding"], tensordict["padding_mask"]
        )

    @staticmethod
    def from_sparse(sparse_embedding: torch.Tensor, batch: torch.Tensor):
        dense, mask = pyg.utils.to_dense_batch(
            sparse_embedding, batch, fill_value=torch.nan
        )
        return ObjectEmbedding(dense_embedding=dense, padding_mask=mask)


def mask_to_batch_indices(mask: torch.Tensor):
    """
    Given the mask indicating which objects are real, return the batch indices.
    This can be seen as the inverse of `torch_geometric.utils.to_dense_batch`.
    batch[i] = j means that the i-th object belongs to the j-th graph.
    :param mask: A boolean mask indicating which object embeddings are real and which are paddings.
        A truthy value means a real embedding at this location.
        Can be with or without batch dimension.
    :return: A tensor of batch indices. shape [N] where N is the number of objects.
    """
    return torch.arange(mask.size(0), device=mask.device).repeat_interleave(
        mask.sum(dim=-1)
    )


class ObjectPoolingModule(torch.nn.Module):

    @staticmethod
    def nansum(dense_embedding: Tensor, _ignored: Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def nanmean(dense_embedding: Tensor, _ignored: Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def nanmax(dense_embedding: Tensor, _ignored: Tensor):
        return dense_embedding.nan_to_num(torch.finfo(dense_embedding.dtype).min).max(
            dim=-2
        )

    def __init__(
        self,
        pooling: Union[str, Callable[[Tensor, Tensor], Tensor]] = "add",
    ) -> None:
        super().__init__()
        self.pooling: Callable[[Tensor, Tensor], Tensor]
        if isinstance(pooling, str):
            if pooling in ("add", "sum"):
                self.pooling = ObjectPoolingModule.nansum
            elif pooling == "mean":
                self.pooling = ObjectPoolingModule.nanmean
            elif pooling == "max":
                self.pooling = ObjectPoolingModule.nanmax
            else:
                raise ValueError(
                    f"Unknown pooling function: {pooling}. Choose from [add, mean, max]."
                )
            self.handles_mask = False
        else:
            self.handles_mask = True
            self.pooling = pooling

    def forward(self, object_embedding: ObjectEmbedding | TensorDict | torch.Tensor):
        """
        Pool the object embeddings.

        Depending on the pooling function, the padding mask may be used to ignore padding embeddings. If the pooling
        function does not support masking, the padding mask is ignored. This is the case with e.g. "sum", "mean", and
        "max" pooling.

        Note: We should reconsider our approach here once pytorch.masked is mature and no longer experimental.
        This applies to other places in the package as well.

        :param object_embedding: The object embeddings to pool.
            Can be a dense tensor, a TensorDict, or an ObjectEmbedding.
            If a TensorDict, it should contain the keys "dense_embedding" and "padding_mask".

        :return: The pooled embeddings.
        """
        if isinstance(object_embedding, torch.Tensor):
            return self.pooling(object_embedding, torch.empty((0,)))

        if isinstance(object_embedding, TensorDict):
            object_embedding = ObjectEmbedding.from_tensordict(object_embedding)

        dense_embedding = object_embedding.dense_embedding
        assert dense_embedding.dim() >= 2

        if self.handles_mask:
            return self.pooling(
                dense_embedding[object_embedding.padding_mask],
                mask_to_batch_indices(object_embedding.padding_mask),
            )
        else:
            return self.pooling(dense_embedding, torch.empty((0,)))
