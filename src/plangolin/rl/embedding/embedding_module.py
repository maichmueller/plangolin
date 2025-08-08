from functools import singledispatchmethod
from typing import List, Sequence

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import Aggregation

import xmimir as xmi
from plangolin.encoding import GraphEncoderBase, HeteroGraphEncoder
from plangolin.models import PyGHeteroModule, PyGModule, RelationalGNN
from plangolin.models.mixins import DeviceAwareMixin
from plangolin.utils.misc import NonTensorWrapper, tolist
from plangolin.utils.object_embeddings import ObjectEmbedding


class EncodingModule(DeviceAwareMixin, torch.nn.Module):
    def __init__(
        self,
        encoder: GraphEncoderBase,
    ):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        items_seq: (
            Sequence[xmi.State] | Sequence[Sequence[xmi.XLiteral]] | NonTensorWrapper
        ),
    ) -> Batch:
        return Batch.from_data_list(
            [
                self.encoder.to_pyg_data(self.encoder.encode(items))
                for items in items_seq
            ]
        ).to(self.device)


class EmbeddingModule(DeviceAwareMixin, torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        encoder: GraphEncoderBase,
        gnn: PyGModule | PyGHeteroModule | None = None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.gnn = gnn
        self.encoding_module = EncodingModule(encoder)

    @singledispatchmethod
    def forward(self, states: List[xmi.State] | NonTensorWrapper) -> ObjectEmbedding:
        states = tolist(states)
        batch = self.encoding_module(states)
        return ObjectEmbedding.from_sparse(*self.gnn(batch))

    @forward.register
    def _(self, states: Batch) -> ObjectEmbedding:
        """This is the case when the states are already in a Batch."""
        return ObjectEmbedding.from_sparse(*self.gnn(states))


def build_hetero_embedding_and_gnn(
    embedding_size: int,
    num_layer: int,
    encoder: HeteroGraphEncoder,
    aggr: str | Aggregation | None = None,
    **kwargs,
):
    gnn = RelationalGNN(
        embedding_size=embedding_size,
        num_layer=num_layer,
        aggr=aggr,
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    return EmbeddingModule(
        embedding_size=gnn.embedding_size, encoder=encoder, gnn=gnn, **kwargs
    )
