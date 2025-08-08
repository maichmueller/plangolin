from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj

from plangolin.encoding.hetero_encoder import PredicateEdgeType
from plangolin.logging_setup import get_logger
from plangolin.models.hetero_message_passing import (
    ConditionalFanOutMP,
    FanInMP,
    FanOutMP,
)
from plangolin.models.logsumexp_aggr import LogSumExpAggregation
from plangolin.models.pyg_module import PyGHeteroModule
from plangolin.utils.object_embeddings import ObjectEmbedding, ObjectPoolingModule

from .mlp import ArityMLPFactory


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


class RelationalGNN(PyGHeteroModule):
    """
    RelationalGNN is a Graph Neural Network designed for learning over relational (heterogeneous) graphs,
    where nodes represent objects and atoms (predicates with arguments), and edges encode relationships
    between them. It performs iterative message passing between objects and atoms, using predicate-specific
    MLPs to aggregate and propagate information. The primary use case is to compute object embeddings in
    planning instances through their atom-relationships.

    Core Functionality:
      - Initializes object and atom embeddings.
      - For each layer, passes messages from objects to atoms (using predicate-specific message modules),
        then from atoms back to objects (using aggregation), and updates object embeddings.
      - Supports customizable aggregation functions, activation functions, and predicate module construction.
      - Handles variable arity predicates and flexible initialization, including random initialization.

    Usage:
      - Instantiate with embedding size, number of layers, object type id, arity dictionary, and optional customizations.
      - Call forward() with x_dict (node features), edge_index_dict (edges), and optional batch or info dicts.
      - Returns object embeddings and their batch indices for downstream tasks.
    """

    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
        predicate_module_factory: Callable[[str, int], torch.nn.Module] | None = None,
        activation: Union[str, Callable, None] = None,
        skip_zero_arity_predicates: bool = True,
        random_init: bool = False,
        random_init_dims: int | None = None,
        random_init_percent: float | None = None,
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
        self.random_initialization: bool = random_init
        self.random_initialization_dims: int
        if random_init_dims is not None:
            self.random_initialization_dims = random_init_dims
        elif random_init_percent is not None:
            self.random_initialization_dims = int(random_init_percent * embedding_size)
        else:
            self.random_initialization_dims = embedding_size
        self.random_initialization_dims = max(
            min(self.random_initialization_dims, embedding_size), 0
        )
        if self.random_initialization_dims == 0:
            get_logger(__name__).warning(
                "Random initialization dimensions are set to 0, random initialization will not be applied."
            )
            self.random_initialization = False

        self.num_layer: int = num_layer
        self.obj_type_id: str = obj_type_id
        if isinstance(aggr, str) or aggr is None:
            if aggr is None or aggr.lower() == "logsumexp":
                aggr = LogSumExpAggregation()
            elif aggr.lower() == "softmax":
                aggr = SoftmaxAggregation()

        activation = activation or "mish"

        # the module to send node-features from objects to atoms
        self.objects_to_atom_mp: torch.nn.Module
        # the module to send created messages from atoms back to objects
        self.atoms_to_object_mp: torch.nn.Module
        # the module to update object embeddings based on the final object messages
        self.embedding_updater: torch.nn.Module
        self._init_modules(
            embedding_size,
            num_layer,
            obj_type_id,
            arity_dict,
            aggr,
            predicate_module_factory,
            activation,
            skip_zero_arity_predicates,
        )

    def _init_modules(
        self,
        embedding_size: int,
        num_layer: int,
        obj_type_id: str,
        arity_dict: Dict[str, int],
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        predicate_module_factory: Callable[[str, int], torch.nn.Module],
        activation: Union[str, Callable, None],
        skip_zero_arity_predicates: bool = True,
    ):
        """
        Initializes the modules for the RelationalGNN.
        """
        if predicate_module_factory is None:
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            predicate_module_factory = ArityMLPFactory(
                feature_size=embedding_size,
                added_arity=0,  # no additional arity
                residual=True,
                padding=None,  # no padding
                layers=1,  # one layer per predicate
                activation=activation,
            )
        predicate_module_dict = {
            pred: predicate_module_factory(pred, arity)
            for pred, arity in arity_dict.items()
            if arity > 0 or not skip_zero_arity_predicates
        }

        self.objects_to_atom_mp = FanOutMP(predicate_module_dict, src_type=obj_type_id)
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

    def initialize_embeddings(
        self, x_dict: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Any]:
        # Initialize embeddings for objects and atoms with 0s.
        # embedding-dims of objects = embedding_size
        # embedding-dims of atoms = (arity of predicate) * embedding_size
        for key, x in x_dict.items():
            assert x.dim() == 2
            init_embed = torch.zeros(
                x.shape[0], x.shape[1] * self.embedding_size, device=x.device
            )
            if self.random_initialization:
                # Random initialization of embeddings
                if self.random_initialization_dims is not None:
                    init_embed = torch.nn.init.xavier_uniform_(
                        init_embed[:, -self.random_initialization_dims :]
                    )
            x_dict[key] = init_embed
        return x_dict, None

    def layer(self, x_dict, edge_index_dict, extra=None):
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
        x_dict, edge_index_dict = self._filter(x_dict, edge_index_dict)

        # Initialize embeddings for objects and atoms.
        x_dict, extra = self.initialize_embeddings(x_dict)

        for _ in range(self.num_layer):
            self.layer(x_dict, edge_index_dict, extra=extra)

        obj_emb = x_dict[self.obj_type_id]
        batch = (
            batch_dict[self.obj_type_id]
            if batch_dict is not None
            else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=obj_emb.device)
        )
        return obj_emb, batch

    def _filter(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.numel() != 0}
        return x_dict, edge_index_dict


class ValueRelationalGNN(RelationalGNN):
    """
    ValueRelationalGNN extends RelationalGNN to provide a value prediction mechanism over graphs.
    After computing object embeddings via relational message passing, it applies a readout MLP and
    a pooling operation to produce a scalar value for each input graph/state.

    Purpose:
      - Useful for tasks such as value estimation in reinforcement learning or evaluating graph-level properties.
      - Aggregates object embeddings into a single value per graph using a configurable pooling strategy.

    Usage:
      - Instantiate like RelationalGNN with additional control over the pooling method.
      - Call forward() to get a tensor of predicted values for each graph in the batch.
    """

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


class ILGRelationalGNN(RelationalGNN):
    """
    ILGRelationalGNN is a variant of RelationalGNN that incorporates an additional condition input
    (e.g., atom status) into each predicate-specific MLP. This allows the GNN to reduce overall MLP count
    by sharing parameters between predicates p and their augmentations (e.g., goal-predicates p_goal,
    [un-]satisfied goal predicates p_sat_goal, ...).

    Differences from RelationalGNN:
      - Increases the embedding size by 1 to accommodate the condition input.
      - Uses ConditionalFanOutMP for message passing with condition-dependent updates.
      - Expects and propagates a condition tensor through the network for each atom.

    Purpose:
      - Identical to RelationalGNN but smaller footprint for scenarios where predicates have multiple variations
        and potential learning benefit from parameter sharing.
    """

    def __init__(self, embedding_size: int, *args, **kwargs):
        # +1 embedding size to add a condition input for each MLP corresponding to the atom status.
        super().__init__(
            embedding_size + 1,
            *args,
            **kwargs,
        )

    def _init_modules(
        self,
        embedding_size: int,
        num_layer: int,
        obj_type_id: str,
        arity_dict: Dict[str, int],
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        predicate_module_factory: Callable[[str, int], torch.nn.Module],
        activation: Union[str, Callable, None],
        skip_zero_arity_predicates: bool = True,
    ):
        """
        Initializes the modules for the RelationalGNN.
        """
        if predicate_module_factory is None:
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            predicate_module_factory = ArityMLPFactory(
                feature_size=embedding_size,
                in_extra_features=1,  # single value feature for the atom status condition
                added_arity=0,  # no additional arity
                residual=True,
                padding=None,  # no padding
                layers=["="],  # one layer per predicate
                activation=activation,
            )
        predicate_module_dict = {
            pred: predicate_module_factory(pred, arity)
            for pred, arity in arity_dict.items()
            if arity > 0 or not skip_zero_arity_predicates
        }

        self.objects_to_atom_mp = ConditionalFanOutMP(
            predicate_module_dict, src_type=obj_type_id
        )
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

    def initialize_embeddings(
        self, x_dict: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        x_dict_copy = dict()
        atom_status_condition = dict()
        for key, x in x_dict.items():
            assert x.dim() == 2
            # remove the last column (it was added extra for 0-arity predicates)
            x_trunc = x[:, :-1]
            if x_trunc.numel() > 0:
                # this model does not support 0-arity predicates, so such atoms are left out
                x_dict_copy[key] = x_trunc
                condition: torch.Tensor = x_trunc.flatten()[0]
                atom_status_condition[key] = condition * torch.ones(
                    (x.shape[0], 1), device=x.device
                )
        x_dict, _ = super().initialize_embeddings(x_dict)
        return x_dict, atom_status_condition

    def layer(self, x_dict, edge_index_dict, extra: dict[str, Tensor] | None = None):
        assert extra is not None, "extra must contain atom-label condition"
        x_dict = {
            k: torch.cat([extra[k], v], dim=1) if k in extra else v
            for k, v in x_dict.items()
        }
        return super().layer(x_dict, edge_index_dict, extra=None)
