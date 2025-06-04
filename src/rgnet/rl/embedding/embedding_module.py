from typing import List

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import Aggregation

import xmimir as xmi
from rgnet.encoding import GraphEncoderBase, HeteroGraphEncoder
from rgnet.models import HeteroGNN, PyGHeteroModule, PyGModule
from rgnet.utils.misc import NonTensorWrapper, tolist
from rgnet.utils.object_embeddings import ObjectEmbedding


class EmbeddingModule(torch.nn.Module):
    def __init__(
        self,
        encoder: GraphEncoderBase,
        gnn: PyGModule | PyGHeteroModule | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.hidden_size = gnn.hidden_size
        self.device = device
        self.gnn = gnn
        self.encoder = encoder

    def forward(self, states: List[xmi.State] | NonTensorWrapper) -> ObjectEmbedding:
        states = tolist(states)
        assert isinstance(states, List)

        as_batch = Batch.from_data_list(
            [self.encoder.to_pyg_data(self.encoder.encode(state)) for state in states]
        )
        as_batch = as_batch.to(self.device)
        return ObjectEmbedding.from_sparse(*self.gnn(as_batch))


def build_hetero_embedding_and_gnn(
    hidden_size: int,
    num_layer: int,
    encoder: HeteroGraphEncoder,
    aggr: str | Aggregation | None = None,
    **kwargs,
):
    gnn = HeteroGNN(
        hidden_size=hidden_size,
        num_layer=num_layer,
        aggr=aggr,
        obj_type_id=encoder.obj_type_id,
        arity_dict=encoder.arity_dict,
    )
    return EmbeddingModule(encoder, gnn=gnn, **kwargs)
