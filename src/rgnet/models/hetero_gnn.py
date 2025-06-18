from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, Optional, Union

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj

from rgnet.encoding.hetero_encoder import PredicateEdgeType
from rgnet.models.hetero_message_passing import FanInMP, FanOutMP
from rgnet.models.logsumexp_aggr import LogSumExpAggregation
from rgnet.models.pyg_module import PyGHeteroModule
from rgnet.models.residual import ResidualModule
from rgnet.utils.object_embeddings import ObjectEmbedding, ObjectPoolingModule


def simple_mlp(
    in_size: int,
    embedding_size: int | Iterable[int],
    out_size: int,
    activation: str | None = None,
):
    activation = activation or "mish"
    if isinstance(embedding_size, Iterable):
        embedding_sizes = tuple(embedding_size)
        embedding_size = embedding_sizes[0]
    else:
        embedding_sizes = (embedding_size,)
    layers = [
        torch.nn.Linear(in_size, embedding_size),
        activation_resolver(activation),
    ]
    for hidden_in_size, hidden_out_size in itertools.pairwise(embedding_sizes):
        layers += [
            torch.nn.Linear(hidden_in_size, hidden_out_size),
            activation_resolver(activation),
        ]
    layers.append(torch.nn.Linear(embedding_sizes[-1], out_size))
    return torch.nn.Sequential(*layers)


class HeteroGNN(PyGHeteroModule):
    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
        activation: Union[str, Callable, None] = None,
    ):
        """
        :param embedding_size: The size of object embeddings.
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

        self.embedding_size: int = embedding_size
        self.num_layer: int = num_layer
        self.obj_type_id: str = obj_type_id
        if isinstance(aggr, str) or aggr is None:
            if aggr is None or aggr.lower() == "logsumexp":
                aggr = LogSumExpAggregation()
            elif aggr.lower() == "softmax":
                aggr = SoftmaxAggregation()

        mlp_dict = {
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            pred: ResidualModule(
                simple_mlp(
                    *([embedding_size * arity] * 3),
                    activation=activation,
                )
            )
            for pred, arity in arity_dict.items()
            if arity > 0
        }

        self.objects_to_atom_mp = FanOutMP(mlp_dict, src_type=obj_type_id)
        self.atoms_to_object_mp = FanInMP(
            embedding_size=embedding_size,
            dst_name=obj_type_id,
            aggr=aggr,
        )
        # Updates object embedding from embedding of last iteration and current iteration:
        # `X_o = comb([X_o, m_o])` where `m_o` is the final object message
        self.embedding_updater = simple_mlp(
            in_size=2 * embedding_size,
            embedding_size=2 * embedding_size,
            out_size=embedding_size,
            activation=activation,
        )

    def initialize_embeddings(self, x_dict: Dict[str, Tensor]):
        # Initialize embeddings for objects and atoms with 0s.
        # embedding-dims of objects = embedding_size
        # embedding-dims of atoms = (arity of predicate) * embedding_size
        for key, x in x_dict.items():
            assert x.dim() == 2
            x_dict[key] = torch.zeros(
                x.shape[0], x.shape[1] * self.embedding_size, device=x.device
            )
        return x_dict

    def layer(self, x_dict, edge_index_dict):
        """
        # Groups object embeddings that are part of an atom and applies predicate-specific Module (e.g. MLP) based on the edge type.
        """
        # Spread the object embeddings to the atoms via message passing.
        # Note: unlike object embeddings, atom embeddings are always simply replaced, instead of updated.
        atom_msgs = self.objects_to_atom_mp(x_dict, edge_index_dict)
        x_dict.update(atom_msgs)

        # Distribute the atom embeddings back to the corresponding objects via message passing.
        object_msgs = self.atoms_to_object_mp(x_dict, edge_index_dict)[self.obj_type_id]
        # perform update step of message passing, but for object-nodes only.
        # The object embeddings are updated based on the previous embedding `X_o` and final object message `m_o`.
        # In formula: `X_o = comb([X_o, m_o])`
        updated_obj_emb = self.embedding_updater(
            torch.cat([x_dict[self.obj_type_id], object_msgs], dim=1)
        )
        # residual update (current + updates)
        x_dict[self.obj_type_id] = x_dict[self.obj_type_id] + updated_obj_emb

    # @torch.compile(fullgraph=True, dynamic=True)
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[PredicateEdgeType, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        info_dict: Optional[Dict[str, Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute object embeddings for each state.
        The states represent graphs and their objects represent nodes.
        The graphs also contain atoms as nodes, but only the object embeddings are returned.
        :param x_dict: The node features for each node type.
            The keys should contain self.obj_type_id.
        :param edge_index_dict: The edges between heterogeneous nodes.
        :param batch_dict: Optional information which node is associated to which state.
            If you pass more than one state (graph) to this function, you should pass the batch_dict too.
        :param info_dict: Optional information about the states.
        :return: A tuple containing:
        - The first tensor contains object embeddings with shape [N, embedding_size], where N is the total number of objects across all states in the batch.
        - The second tensor contains batch indices with shape [N], mapping each object (node) to its corresponding state (graph).
        Note that the number of objects is not necessarily equal for each state. This tuple can be used to instantiate an `ObjectEmbedding`.
        We do not return this object directly since pytorch would refuse to accept hooks with this return type.
        """
        # Filter out dummies
        x_dict = {k: v for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.numel() != 0}

        # Resize everything by the embedding_size
        self.initialize_embeddings(x_dict)

        for _ in range(self.num_layer):
            self.layer(x_dict, edge_index_dict)

        obj_emb = x_dict[self.obj_type_id]
        batch = (
            batch_dict[self.obj_type_id]
            if batch_dict is not None
            else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=obj_emb.device)
        )
        return obj_emb, batch


class ValueHeteroGNN(HeteroGNN):
    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        obj_type_id: str,
        arity_dict: Dict[str, int],
        aggr: Optional[str | pyg.nn.aggr.Aggregation] = None,
        activation: Union[str, Callable, None] = None,
        pooling: Union[str, Callable[[Tensor], Tensor]] = "add",
    ):
        super().__init__(
            embedding_size,
            num_layer,
            aggr,
            obj_type_id,
            arity_dict,
            activation=activation,
        )
        self.readout = simple_mlp(
            embedding_size, 2 * embedding_size, 1, activation=activation
        )
        self.pooling = ObjectPoolingModule(pooling)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[PredicateEdgeType, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        info_dict: Optional[Dict[str, Tensor]] = None,
    ) -> torch.Tensor:
        object_embeddings = ObjectEmbedding.from_sparse(
            *super().forward(x_dict, edge_index_dict, batch_dict)
        )
        return self.readout(self.pooling(object_embeddings)).view(-1)
