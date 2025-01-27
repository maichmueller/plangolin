from typing import List

import pymimir as mi
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import Aggregation

from rgnet.encoding import HeteroGraphEncoder
from rgnet.models import HeteroGNN
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, tolist
from rgnet.utils.object_embeddings import ObjectEmbedding


class EmbeddingModule(torch.nn.Module):

    def __init__(
        self,
        encoder: HeteroGraphEncoder,
        gnn: HeteroGNN | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.hidden_size = gnn.hidden_size
        self.device = device
        self.gnn = gnn
        self.encoder: HeteroGraphEncoder = encoder

    def forward(self, states: List[mi.State] | NonTensorWrapper) -> ObjectEmbedding:
        states = tolist(states)
        assert isinstance(states, List)

        as_batch = Batch.from_data_list(
            [self.encoder.to_pyg_data(self.encoder.encode(state)) for state in states]
        )
        as_batch = as_batch.to(self.device)
        return self.gnn(as_batch.x_dict, as_batch.edge_index_dict, as_batch.batch_dict)


def build_embedding_and_gnn(
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
