from typing import Callable, Dict, Optional, Union

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.typing import Adj

from rgnet.encoding.hetero_encoder import PredicateEdgeType
from rgnet.models.hetero_message_passing import FanInMP, FanOutMP
from rgnet.utils.object_embeddings import ObjectEmbedding, ObjectPoolingModule


def simple_mlp(in_size: int, hidden_size: int, out_size: int, **kwargs):
    if "act" not in kwargs:
        kwargs["act"] = "mish"
    channel_list = [in_size, hidden_size, hidden_size]
    if out_size != hidden_size:
        channel_list.append(out_size)

    return pyg.nn.MLP(channel_list, norm=None, dropout=0.0, **kwargs)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        activation: Union[str, Callable, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.mlp = simple_mlp(hidden_size, hidden_size, hidden_size, act=activation)

    def forward(self, input_tensor: torch.Tensor):
        return input_tensor + self.mlp(input_tensor)


class HeteroGNN(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
        activation: Union[str, Callable, None] = None,
    ):
        """
        :param hidden_size: The size of object embeddings.
        :param num_layer: Total number of message exchange iterations.
        :param aggr: Aggregation-function to be used for message passing.
        :param obj_type_id: The type identifier of objects in the x_dict.
        :param arity_dict: A dictionary mapping predicates names to their arity.
        :param activation: The activation function for all MLPs
            (Default mish).
        Creates one MLP for each predicate.
        Note that predicates as well as goal-predicates are meant.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layer: int = num_layer
        self.obj_type_id: str = obj_type_id
        if aggr == "softmax":
            aggr = SoftmaxAggregation()

        mlp_dict = {
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            pred: ResidualBlock(hidden_size * arity, activation=activation)
            for pred, arity in arity_dict.items()
            if arity > 0
        }

        self.obj_to_atom = FanOutMP(mlp_dict, src_name=obj_type_id)

        # Updates object embedding from embedding of last iteration and current iteration.
        self.obj_update = simple_mlp(
            in_size=2 * hidden_size,
            hidden_size=2 * hidden_size,
            out_size=hidden_size,
            act=activation,
        )
        # Messages from atoms flow to objects
        self.atom_to_obj = FanInMP(
            hidden_size=hidden_size,
            dst_name=obj_type_id,
            aggr=aggr,
        )

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

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[PredicateEdgeType, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ) -> ObjectEmbedding:
        """
        Compute object embeddings for each state.
        The states represent graphs and their objects represent nodes.
        The graphs also contain atoms as nodes, but only the object embeddings are returned.
        :param x_dict: The node features for each node type.
            The keys should contain self.obj_type_id.
        :param edge_index_dict: The edges between heterogeneous nodes.
        :param batch_dict: Optional information which node is associated to which state.
            If you pass more than one state (graph) to this function, you should pass the batch_dict too.
        :return: A tuple containing:
        - The first tensor contains object embeddings with shape [N, hidden_size], where N is the total number of objects across all states in the batch.
        - The second tensor contains batch indices with shape [N], mapping each object (node) to its corresponding state (graph).
        Note that the number of objects is not necessarily equal for each state.
        """
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
        return ObjectEmbedding.from_sparse(obj_emb, batch)


class ValueHeteroGNN(HeteroGNN):

    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
        activation: Union[str, Callable, None] = None,
        pooling: Union[str, Callable[[Tensor, Tensor], Tensor]] = "add",
    ):
        super().__init__(
            hidden_size,
            num_layer,
            aggr,
            obj_type_id,
            arity_dict,
            activation=activation,
        )
        self.readout = simple_mlp(hidden_size, 2 * hidden_size, 1, act=activation)
        self.pooling = ObjectPoolingModule(pooling)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[PredicateEdgeType, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ) -> torch.Tensor:
        object_embeddings = super().forward(x_dict, edge_index_dict, batch_dict)
        return self.readout(self.pooling(object_embeddings)).view(-1)
