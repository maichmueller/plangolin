from typing import List

import pymimir as mi
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import Aggregation

from rgnet import HeteroGNN, HeteroGraphEncoder
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, non_tensor_to_list


class EmbeddingModule(torch.nn.Module):

    def __init__(
        self,
        encoder: HeteroGraphEncoder,
        hidden_size: int,
        num_layer: int,
        aggr: str | Aggregation | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.gnn = HeteroGNN(
            hidden_size=hidden_size,
            num_layer=num_layer,
            aggr=aggr,
            obj_type_id=encoder.obj_type_id,
            arity_dict=encoder.arity_dict,
        )
        self.encoder: HeteroGraphEncoder = encoder

    def forward(self, states: List[mi.State] | NonTensorWrapper) -> torch.Tensor:
        states = non_tensor_to_list(states)
        assert isinstance(states, List)

        as_batch = Batch.from_data_list(
            [self.encoder.to_pyg_data(self.encoder.encode(state)) for state in states]
        )
        # TODO send to device?
        return self.gnn.calculate_embedding(
            as_batch.x_dict, as_batch.edge_index_dict, as_batch.batch_dict
        )
