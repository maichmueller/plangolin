from __future__ import annotations

from typing import Callable, List, Literal, Union

import torch
import torch_geometric as pyg
from tensordict import TensorDict
from torch import Tensor

from plangolin.logging_setup import get_logger
from plangolin.utils.reshape import unsqueeze_right, unsqueeze_right_like


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
        at object-index `o` of batch-element `n` is a real (True) or padded embedding (False).

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

    def tensor_split(self, split_indices_tensor: torch.Tensor) -> List[ObjectEmbedding]:
        if not split_indices_tensor.is_cpu:
            get_logger(__name__).info(
                "Received split tensor that was on the GPU."
                " Torch requires it to be on the CPU."
                " Performing implicit device move."
            )
            split_indices_tensor = split_indices_tensor.cpu()
        split_dense = self.dense_embedding.tensor_split(split_indices_tensor)
        split_mask = self.padding_mask.tensor_split(split_indices_tensor)
        return [
            ObjectEmbedding(dense, mask) for dense, mask in zip(split_dense, split_mask)
        ]

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

    def to(self, device: torch.device) -> ObjectEmbedding:
        """
        Move the object embedding to the specified device.
        """
        self.dense_embedding = self.dense_embedding.to(device)
        self.padding_mask = self.padding_mask.to(device)
        return self

    def cpu(self):
        """
        Move the object embedding to the CPU.
        """
        self.dense_embedding = self.dense_embedding.cpu()
        self.padding_mask = self.padding_mask.cpu()
        return self

    def clone(self) -> ObjectEmbedding:
        """
        Clone the object embedding.
        """
        # shortcut the __init__ logic which only verifies the tensor shapes needlessly in this case
        obj = ObjectEmbedding.__new__(ObjectEmbedding)
        obj.dense_embedding = self.dense_embedding.clone()
        obj.padding_mask = self.padding_mask.clone()
        return obj

    def detach(self) -> ObjectEmbedding:
        """
        Detach the object embedding from the computation graph.
        """
        return ObjectEmbedding(
            self.dense_embedding.detach(),
            self.padding_mask.detach(),
        )

    def allclose(
        self,
        other: TensorDict | ObjectEmbedding,
    ) -> bool:
        if isinstance(other, TensorDict):
            other = ObjectEmbedding.from_tensordict(other)
        return (
            torch.allclose(self.dense_embedding, other.dense_embedding)
            and (self.padding_mask == other.padding_mask).all()
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
    """
    Global pooling-functor of object embeddings.

    Global means with respect to all states in a batch.
    Depending on the pooling function, the padding mask may be used to ignore padding embeddings. If the pooling
    function does not support masking, the padding mask is ignored. This is the case with e.g. "sum", "mean", and
    "max" pooling.
    """

    options = ["sum", "mean", "max", "add"]

    @staticmethod
    def sum(dense_embedding: torch.Tensor):
        return ObjectPoolingModule.add(dense_embedding)

    @staticmethod
    def add(dense_embedding: torch.Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def mean(dense_embedding: torch.Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def max(dense_embedding: torch.Tensor):
        return dense_embedding.nan_to_num(torch.finfo(dense_embedding.dtype).min).max(
            dim=-2
        )

    def __init__(
        self,
        pooling: Union[
            Literal["sum", "mean", "max", "add"],
            Callable[[Tensor], Tensor],
        ] = "sum",
    ) -> None:
        super().__init__()
        self.pooling: Callable[[Tensor], Tensor]

        if isinstance(pooling, str):
            pooling = self._verify_pool_str(pooling)
            self.pooling = getattr(ObjectPoolingModule, pooling)
        else:
            self.raw_tensor_based = True
            self.pooling = pooling

    def _verify_pool_str(self, pooling):
        if pooling not in ObjectPoolingModule.options:
            raise ValueError(
                f"Invalid pooling function: {pooling}. Choose from {self.options}."
            )
        return pooling.replace("sum", "add")

    def forward(self, object_embedding: ObjectEmbedding | TensorDict | torch.Tensor):
        if isinstance(object_embedding, torch.Tensor):
            return self.pooling(object_embedding)
        if isinstance(object_embedding, TensorDict):
            object_embedding = ObjectEmbedding.from_tensordict(object_embedding)
        dense_embedding = object_embedding.dense_embedding
        assert dense_embedding.dim() >= 2
        return self.pooling(dense_embedding)
