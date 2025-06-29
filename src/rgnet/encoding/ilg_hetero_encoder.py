from __future__ import annotations

import logging
import operator
from collections import defaultdict
from enum import Enum
from functools import cache
from itertools import chain
from typing import Dict, Iterable, List, NamedTuple, Sequence

import networkx as nx
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from xmimir import XAtom, XDomain, XLiteral, XPredicate, atom_str_template

from .base_encoder import GraphEncoderBase, GraphT, check_encoded_by_this
from .node_factory import Node, NodeFactory


class PredicateEdgeType(NamedTuple):
    src_type: str
    pos: str
    dst_type: str


class AtomStatus(Enum):
    REGULAR = 0
    GOAL = 1
    SATISFIED_GOAL = 2
    UNSATISFIED_GOAL = 3
    SATISFIED_NEGATED_GOAL = 4
    UNSATISFIED_NEGATED_GOAL = 5


class HeteroILGGraphEncoder(GraphEncoderBase[nx.MultiGraph]):
    """
    An encoder to represent states as heterogeneous graphs with objects and atoms as vertices
    and edges (i, j) whenever an atom p(..., i, j, ...) holds in the state.

    The heterogeneous aspect is that there are different types of nodes:
    object nodes and atom nodes of a number of different predicate types.
    Objects are connected only to atom nodes, while atom nodes are connected to object nodes in turn.
    Merely, arity 0 predicates are connected to themselves, i.e. they have a self-loop to retain representation (e.g.,
    for GNNs that consider globally aggregating nodes).

    Follows the Instance-Learning Graph (ILG) approach[1]. In particular, each atom receives a colour (feature)
    depending on its status (see `AtomStatus` enum) and no separate goal atoms are created.

    References:
    [1] https://arxiv.org/abs/2403.16508
    """

    def __init__(
        self,
        domain: XDomain,
        node_factory: NodeFactory | None = None,
        obj_type_id: str = "obj",
    ) -> None:
        super().__init__(domain=domain)
        self.obj_type_id: str = obj_type_id
        # Initialize the default here to please jsonargparse.
        self.node_factory: NodeFactory = node_factory or NodeFactory()
        self.predicates: tuple[XPredicate, ...] = self.domain.predicates()
        self.arity_dict: Dict[Node, int] = self.make_arity_dict(
            self.predicates, self.node_factory
        )
        # Generate all possible edge types
        self.all_edge_types: List[EdgeType] = []
        for predicate, arity in self.arity_dict.items():
            for pos in range(arity):
                self.all_edge_types.append((self.obj_type_id, str(pos), predicate))
                self.all_edge_types.append((predicate, str(pos), self.obj_type_id))
        self.arity_dict = {
            pred: arity
            for pred, arity in self.arity_dict.items()
            if not pred.endswith(self.node_factory.goal_suffix)
        }

    def __eq__(self, other: HeteroILGGraphEncoder) -> bool:
        if not isinstance(other, HeteroILGGraphEncoder):
            return NotImplemented
        return (
            self.obj_type_id == other.obj_type_id
            and self.node_factory == other.node_factory
            and self.predicates == other.predicates
        )

    @staticmethod
    @cache
    def make_arity_dict(
        predicates: Iterable[XPredicate], node_factory: NodeFactory = NodeFactory()
    ) -> Dict[Node, int]:
        return dict(
            sorted(
                (
                    (node_factory(predicate), predicate.arity)
                    for predicate in predicates
                ),
                key=operator.itemgetter(0),
            )
        )

    def _encode(self, items: Sequence[XAtom | XLiteral], graph: GraphT):
        # Build hetero graph from state
        # One node for each object
        # One node for each atom
        # Edge label = position in atom

        actual_atoms, missing_goal_atoms, statuses = self._compute_statuses(items)

        for obj in self._contained_objects(items):
            graph.add_node(
                self.node_factory(obj), type=self.obj_type_id, name=obj.get_name()
            )

        for atom in chain(actual_atoms, missing_goal_atoms):
            predicate = atom.predicate
            atom_node = self.node_factory(atom)
            graph.add_node(
                atom_node,
                type=self.node_factory(predicate),
                status=statuses[atom],
            )
            if predicate.arity == 0:
                # If the predicate has arity 0, we add a self-loop to the atom node to ensure it  is connected
                graph.add_edge(atom_node, atom_node, position=-1)
            else:
                for pos, obj in enumerate(atom.objects):
                    # Connect atom node to object nodes
                    graph.add_edge(self.node_factory(obj), atom_node, position=pos)

    def _compute_statuses(self, items):
        goals: list[XLiteral] = []
        actual_atoms: list[XAtom] = []
        for atom_or_literal in items:
            if isinstance(atom_or_literal, XLiteral):
                if getattr(atom_or_literal, "is_not_goal", False):
                    raise ValueError(
                        "XLiteral should not be marked as a non-goal. Only goal literals are supported."
                    )
                goals.append(atom_or_literal)
            else:
                actual_atoms.append(atom_or_literal)
        goal_matches = dict()
        for atom in actual_atoms:
            for goal in goals:
                if goal.atom.semantic_eq(atom):
                    goal_matches[atom] = goal
                    goal_matches[goal] = atom
                    break
        statuses: dict[XAtom, AtomStatus] = defaultdict(lambda: AtomStatus.REGULAR)
        # Add goal atoms to the graph that are not matched by any actual atom in the graph. These are positive goal
        # atoms that are either unsatisfied or satisfied negated goals. In both cases the atom is not true in the state,
        # and thus missing. To encode this information we add it in extra.
        missing_goal_atoms: list[XAtom] = []
        for goal in goals:
            atom = goal.atom
            if goal in goal_matches:
                if goal.is_negated:
                    statuses[atom] = AtomStatus.UNSATISFIED_NEGATED_GOAL
                else:
                    statuses[atom] = AtomStatus.SATISFIED_GOAL
            else:
                missing_goal_atoms.append(atom)
                if goal.is_negated:
                    statuses[atom] = AtomStatus.SATISFIED_NEGATED_GOAL
                else:
                    statuses[atom] = AtomStatus.UNSATISFIED_GOAL
        return actual_atoms, missing_goal_atoms, statuses

    @check_encoded_by_this
    def to_pyg_data(self, graph: GraphT) -> HeteroData:
        del graph.graph["encoding"]
        del graph.graph["state"]
        nodes_dict: Dict[NodeType, List[Node]] = defaultdict(list)
        object_names: List[str] = []
        for node, node_data in graph.nodes(data=True):
            ntype = node_data["type"]
            nodes_dict[ntype].append(node)
            if ntype == self.obj_type_id:
                object_names.append(node_data["name"])

        node_idx_dict: Dict[NodeType, Dict[Node, int]] = {
            ntype: {node: i for i, node in enumerate(nodes)}
            for ntype, nodes in nodes_dict.items()
        }

        data = HeteroData()
        data.object_names = object_names
        # Create x_dict, the feature matrix for each node type
        # Our features for atom-nodes are their AtomStatus, so we just create a uniform tensor of that enum value of
        # size `arity`.
        # For object nodes, we use a uniform feature vector of size 1.
        for node_type, nodes_of_type in nodes_dict.items():
            if node_type == self.obj_type_id:
                # we give x two dimensions of features here (instead of just 1), to make it easier to later simply call
                # x.shape[1] - 1 to get the arity of the atom without accidentally ruining object initialization
                x = torch.zeros((len(nodes_of_type), 2), dtype=torch.float32)
            else:
                arity = self.arity_dict[node_type]
                # ensure arity 0 also has a non-empty status vector
                x = torch.empty((len(nodes_of_type), arity + 1), dtype=torch.float32)
                for i, node in enumerate(nodes_of_type):
                    status: AtomStatus = graph.nodes[node]["status"]
                    x[i].fill_(status.value)
            data[node_type].x = x

        # Add dummy entry for node-types that don't appear in this state
        # https://github.com/pyg-team/pytorch_geometric/issues/9233
        for unused_node_type in self.arity_dict.keys() - nodes_dict.keys():
            data[unused_node_type].x = torch.empty(0, dtype=torch.float32)
        if self.obj_type_id not in nodes_dict:
            logging.warning(f"No object in graph ({graph})")
            data[self.obj_type_id].x = torch.empty(0, dtype=torch.float32)

        # Group edges by src, position, dst
        edge_dict: Dict[EdgeType, List[torch.Tensor]] = defaultdict(list)
        for src, dst, attr in graph.edges.data():
            src_type: str = graph.nodes[src]["type"]
            dst_type: str = graph.nodes[dst]["type"]
            pos = str(attr["position"])

            forward_edge_type = PredicateEdgeType(src_type, pos, dst_type)
            reverse_edge_type = PredicateEdgeType(dst_type, pos, src_type)

            for edge_type, (src_node_idx, dst_node_idx) in (
                (forward_edge_type, (src, dst)),
                (reverse_edge_type, (dst, src)),
            ):
                edge_dict[edge_type].append(
                    torch.tensor(
                        (
                            node_idx_dict[edge_type.src_type][src_node_idx],
                            node_idx_dict[edge_type.dst_type][dst_node_idx],
                        )
                    )
                )

        # Stack grouped edge_indices and add to data
        for edge_type, node_type in edge_dict.items():
            # HeteroData want Tuple[str,str,str] as edge keys
            data[edge_type].edge_index = torch.stack(node_type, dim=1)

        # Add dummy entry for edge-types that don't appear in this state
        for unused_edge_type in self.all_edge_types - edge_dict.keys():
            data[unused_edge_type].edge_index = torch.empty(2, 0, dtype=torch.long)

        return data

    def from_pyg_data(self, data: HeteroData) -> GraphT:
        """
        Reconstruct a graph from a HeteroData object.
        Every node has a type attribute, either self.obj_type_id or a predicate name.
        Node names are the concatenation of the type and the index in the feature matrix.
        Edge labels are the position in the atom.
        The returned graph is not the exact same as one returned by encode, but isomorphic.
        :param data: HeteroData object encoded with this encoder.
        :return: The networkx graph as described above.
        """
        graph = self._graph_t(encoding=self)
        assert all(pred in data.node_types for pred in self.arity_dict.keys())
        obj_type: str = self.obj_type_id
        # Every node needs a unique name, but obj-names are lost, therefore we use the
        # index in the feature matrix together with the type.
        obj_names = data.object_names
        for obj_idx in range(data.x_dict[obj_type].shape[0]):
            graph.add_node(obj_names[obj_idx], type=obj_type, name=obj_names[obj_idx])

        edge_types = set()
        predicates = set()
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            predicates.add(src)
            predicates.add(dst)
        predicates.discard(self.obj_type_id)
        if predicates - self.arity_dict.keys():
            raise ValueError(
                f"Graph predicates are not a subset of the encoder's domain predicates: "
                f"({predicates} != {self.arity_dict.keys()})"
            )
        atoms = defaultdict(dict)
        for edge_type in data.edge_types:
            if edge_type in edge_types:
                raise ValueError(f"Duplicate edge type found ({edge_type}).")
            if edge_type in edge_types:
                continue
            edge_types.add(edge_type)
            src_tensor, dst_tensor = data.edge_index_dict[edge_type]
            src, rel, dst = edge_type
            for src_idx, dst_idx in zip(src_tensor, dst_tensor):
                src_idx = src_idx.item()
                dst_idx = dst_idx.item()
                if src != self.obj_type_id:
                    src_status = data.x_dict[src].flatten()[0].item()
                    atoms[(src, src_idx, src_status)][int(rel)] = (
                        obj_names[dst_idx] if dst_idx >= 0 else None
                    )
                elif dst != self.obj_type_id:
                    dst_status = data.x_dict[dst].flatten()[0].item()
                    atoms[(dst, dst_idx, dst_status)][int(rel)] = (
                        obj_names[src_idx] if src_idx >= 0 else None
                    )
                else:
                    raise ValueError(
                        f"Both src and dst are objects: {edge_type=}, {src_idx=}, {dst_idx=}."
                    )
        for (predicate, atom_idx, status), obj_dict in atoms.items():
            pos_objname_tuples = list(
                filter(
                    lambda x: x[0] > -1,
                    sorted(obj_dict.items(), key=operator.itemgetter(0)),
                )
            )
            atom_str = atom_str_template.render(
                predicate=predicate,
                objects=[obj_name for pos, obj_name in pos_objname_tuples],
            )
            graph.add_node(atom_str, type=predicate, status=AtomStatus(status))
            if pos_objname_tuples:
                for pos, obj_name in pos_objname_tuples:
                    graph.add_edge(
                        atom_str,
                        obj_name,
                        position=int(pos),
                    )
            else:
                # self-loop for arity 0 predicates
                graph.add_edge(
                    atom_str,
                    atom_str,
                    position=-1,
                )
        return self._graph_t(graph)
