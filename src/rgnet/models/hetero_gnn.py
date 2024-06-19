from typing import Dict, Optional

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.typing import Adj

from rgnet.models.hetero_message_passing import FanInMP, FanOutMP


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
    ):
        """
        :param hidden_size: The size of object embeddings.
        :param num_layer: Total number of message exchange iterations.
        :param aggr: Aggregation function to be used for message passing.
        :param obj_type_id: The type identifier of objects in the x_dict.
        :param arity_dict: A dictionary mapping predicate names to their arity.
        Creates one MLP for each predicate.
        Note that predicates as well as goal-predicates are meant.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layer: int = num_layer
        self.obj_type_id: str = obj_type_id
        mlp_dict = {
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            pred: HeteroGNN.mlp(
                hidden_size * arity, hidden_size * arity, hidden_size * arity
            )
            for pred, arity in arity_dict.items()
            if arity > 0
        }

        self.obj_to_atom = FanOutMP(mlp_dict, src_name=obj_type_id)

        self.obj_update = HeteroGNN.mlp(
            in_size=2 * hidden_size, hidden_size=2 * hidden_size, out_size=hidden_size
        )
        # Messages from atoms flow to objects
        self.atom_to_obj = FanInMP(
            hidden_size=hidden_size,
            dst_name=obj_type_id,
            aggr=aggr,
        )
        self.readout = HeteroGNN.mlp(hidden_size, 2 * hidden_size, 1)

    def encoding_layer(self, x_dict: Dict[str, Tensor]):
        # Resize everything by the hidden_size
        # embedding of objects = hidden_size
        # embedding of atoms = arity of predicate * hidden_size
        for k, v in x_dict.items():
            assert v.dim() == 2
            x_dict[k] = torch.zeros(
                v.shape[0], v.shape[1] * self.hidden_size, device=v.device
            )
        return x_dict

    def layer(self, x_dict, edge_index_dict):
        # Groups object embeddings that are part of an atom and
        # applies predicate-specific MLP based on the edge type.
        out = self.obj_to_atom(x_dict, edge_index_dict)
        x_dict.update(out)  # update atom embeddings
        # Distribute the atom embeddings back to the corresponding objects.
        out = self.atom_to_obj(x_dict, edge_index_dict)
        # Update the object embeddings using a shared update-MLP.
        obj_emb = torch.cat([x_dict[self.obj_type_id], out[self.obj_type_id]], dim=1)
        obj_emb = self.obj_update(obj_emb)
        x_dict[self.obj_type_id] = obj_emb

    def calculate_embedding(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):
        # Filter out dummies
        x_dict = {k: v for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.numel() != 0}

        x_dict = self.encoding_layer(x_dict)  # Resize everything by the hidden_size

        for _ in range(self.num_layer):
            self.layer(x_dict, edge_index_dict)

        obj_emb = x_dict[self.obj_type_id]
        batch = (
            batch_dict[self.obj_type_id]
            if batch_dict is not None
            else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=obj_emb.device)
        )
        # Aggregate all object embeddings into one aggregated embedding
        return pyg.nn.global_add_pool(obj_emb, batch)  # shape [hidden, 1]

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):
        aggr = self.calculate_embedding(x_dict, edge_index_dict, batch_dict)
        # Produce final single scalar of shape [1]
        return self.readout(aggr).view(-1)

    @staticmethod
    def mlp(in_size: int, hidden_size: int, out_size: int):
        return pyg.nn.MLP([in_size, hidden_size, out_size], norm=None, dropout=0.0)
