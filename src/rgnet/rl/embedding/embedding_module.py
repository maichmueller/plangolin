from functools import singledispatchmethod
from typing import List, Sequence

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import Aggregation

import xmimir as xmi
from rgnet.encoding import GraphEncoderBase, HeteroGraphEncoder
from rgnet.models import HeteroGNN, PyGHeteroModule, PyGModule
from rgnet.models.mixins import DeviceAwareMixin
from rgnet.utils.misc import NonTensorWrapper, tolist
from rgnet.utils.object_embeddings import ObjectEmbedding


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


class EmbeddingModule(torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        encoder: GraphEncoderBase,
        gnn: PyGModule | PyGHeteroModule | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.gnn = gnn
        self.encoder = encoder

    @singledispatchmethod
    def forward(self, states: List[xmi.State] | NonTensorWrapper) -> ObjectEmbedding:
        states = tolist(states)

        as_batch = Batch.from_data_list(
            [self.encoder.to_pyg_data(self.encoder.encode(state)) for state in states]
        )
        as_batch = as_batch.to(self.device)
        return ObjectEmbedding.from_sparse(*self.gnn(as_batch))

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
    gnn = HeteroGNN(
        embedding_size=embedding_size,
        num_layer=num_layer,
        aggr=aggr,
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    return EmbeddingModule(
        embedding_size=gnn.embedding_size, encoder=encoder, gnn=gnn, **kwargs
    )
