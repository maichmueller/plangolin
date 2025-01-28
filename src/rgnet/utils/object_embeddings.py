from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch_geometric as pyg
from tensordict import TensorDict
from torch import Tensor
from torch.masked import MaskedTensor

from rgnet.utils.reshape import unsqueeze_right, unsqueeze_right_like


class ObjectEmbedding:
    dense_embedding: Tensor
    padding_mask: Tensor

    def __init__(
        self,
        dense_embedding: Tensor,
        padding_mask: Tensor,
    ) -> None:
        """
        An abstraction for object embeddings of a batch with potentially different numbers of objects per element.

        These masks are created when using `torch_geometric`'s `to_dense_batch` function.
        If all batch-elements have the same number of objects, the padding mask is a
        boolean tensor of shape [N, O] with all True values.
        If the number of objects varies, the padding mask is a boolean tensor of shape [N, O]
        with False values at the padded locations.
        A padding mask M highlights at entry M[n, o] whether the object embedding
        at object-index `o` of batch-element `n` is a padded embedding or a real one.

        :param dense_embedding: The dense object embeddings.
            Shape: [N, O, (F, ...)], where...
                N is the number elements (states) in a batch,
                O is the number of objects,
                F is the embedding feature shape (e.g. a single dimension of size F).
        :param padding_mask: A boolean mask indicating which object embeddings are real and which are paddings.
            A truthy value means a real embedding at this location.
            Shape: [N, O] with dtype bool.
        """
        padding_dim = padding_mask.ndim
        if (dim_diff := 2 - padding_dim) > 0:
            padding_mask = unsqueeze_right(padding_mask, dim_diff)
        embedding_dim = dense_embedding.ndim
        if embedding_dim < 2:
            raise ValueError(
                f"Embeddings tensor must have at least 2 dimensions. Got {dense_embedding.shape}."
            )
        elif embedding_dim == 2:
            dense_embedding = dense_embedding.unsqueeze(1)
        self.dense_embedding = dense_embedding
        self.padding_mask = padding_mask
        self._assert_dim_match(0, 1)

    def _assert_dim_match(self, *dims: int):
        dense_embedding = self.dense_embedding
        padding_mask = self.padding_mask
        for dim in dims:
            if dense_embedding.size(dim) != padding_mask.size(dim):
                raise ValueError(
                    f"Sizes of 'dense_embedding' and 'padding_mask' at dimension {dim} do not match: "
                    f"{dense_embedding.size(dim)} != {padding_mask.size(dim)}"
                )

    def __eq__(self, other: object):
        if not isinstance(other, ObjectEmbedding):
            return NotImplemented
        return torch.equal(
            self.dense_embedding[self.padding_mask],
            other.dense_embedding[other.padding_mask],
        )

    def __repr__(self):
        return f"ObjectEmbedding(dense_embedding={self.dense_embedding}, padding_mask={self.padding_mask})"

    def to_tensordict(self) -> TensorDict:
        return TensorDict(
            {
                "dense_embedding": self.dense_embedding,
                "padding_mask": self.padding_mask,
            },
            batch_size=(self.dense_embedding.size(0),),
        )

    def to_masked_tensor(self):
        """
        Convert the object_embedding to a masked tensor.
        """
        unsqueezed_mask = unsqueeze_right_like(self.padding_mask, self.dense_embedding)
        return torch.masked.masked_tensor(
            self.dense_embedding,
            unsqueezed_mask.expand_as(self.dense_embedding),
            requires_grad=self.dense_embedding.requires_grad,
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
    def sum(dense_embedding: Tensor):
        return torch.sum(dense_embedding, dim=-2)

    @staticmethod
    def add(dense_embedding: Tensor):
        return torch.sum(dense_embedding, dim=-2)

    @staticmethod
    def mean(dense_embedding: Tensor):
        return torch.mean(dense_embedding, dim=-2)

    @staticmethod
    def max(dense_embedding: Tensor):
        return torch.amax(dense_embedding, dim=-2)

    def __init__(
        self,
        pooling: Union[
            str,
            Callable[[MaskedTensor], Tensor],
            Callable[[Tensor, Tensor], Tensor],
        ] = "add",
    ) -> None:
        super().__init__()
        if isinstance(pooling, str):
            options = ["sum", "mean", "max", "add"]
            if pooling not in options:
                raise ValueError(
                    f"Invalid pooling function: {pooling}. Choose from {options}."
                )
            self.pooling = vars(ObjectPoolingModule)[pooling]
            self.raw_tensor_based = False
        else:
            self.raw_tensor_based = True
            self.pooling = pooling

    def forward(self, object_embedding):
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
            object_embedding = torch.masked.masked_tensor(
                object_embedding, torch.isnan(object_embedding)
            )

        if isinstance(object_embedding, MaskedTensor):
            assert self.raw_tensor_based
            return self.pooling(object_embedding)

        if isinstance(object_embedding, TensorDict):
            object_embedding = ObjectEmbedding.from_tensordict(object_embedding)

        dense_embedding = object_embedding.dense_embedding
        assert dense_embedding.dim() >= 2
        return self.pooling(
            dense_embedding[object_embedding.padding_mask],
            mask_to_batch_indices(object_embedding.padding_mask),
        )
