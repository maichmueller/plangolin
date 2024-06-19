from typing import List, Tuple

import pymimir as mi
import torch
from tensordict import NonTensorStack
from tensordict.nn import TensorDictModule
from torch_geometric.data import Batch
from torch_geometric.nn import Aggregation

from rgnet import HeteroGNN, HeteroGraphEncoder
from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


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

    def forward(self, states: List[mi.State]) -> torch.Tensor:
        assert isinstance(states, List)

        as_batch = Batch.from_data_list(
            [self.encoder.to_pyg_data(self.encoder.encode(state)) for state in states]
        )
        # TODO send to device?
        return self.gnn.calculate_embedding(
            as_batch.x_dict, as_batch.edge_index_dict, as_batch.batch_dict
        )


def embed_states_and_transitions(
    embedding: EmbeddingModule,
    states_key: str,
    transitions_key: str,
    out_keys: List[str],
) -> TensorDictModule:
    """
    Use the embedding
    :rtype: object
    """
    assert len(out_keys) == 2

    def forward(
        states: List[mi.State] | NonTensorWrapper,
        transitions_list: List[List[mi.Transition]] | NonTensorWrapper,
    ) -> Tuple[torch.Tensor, NonTensorStack]:  # second entry is List[torch.Tensor]

        states = non_tensor_to_list(states)
        transitions_list = non_tensor_to_list(transitions_list)

        assert isinstance(states, List) and isinstance(transitions_list, List)

        current_embedding_batch = embedding(states)
        # We can't create a tensor because the number of successors can vary between
        # each state in the batch.
        successor_embedding_batch = as_non_tensor_stack(
            [
                embedding([t.target for t in transitions])
                for transitions in transitions_list
            ]
        )
        return current_embedding_batch, successor_embedding_batch

    return TensorDictModule(
        forward, in_keys=[states_key, transitions_key], out_keys=out_keys
    )
