from __future__ import annotations

import itertools
import logging
import operator
from collections import defaultdict
from functools import cache
from typing import Dict, Iterable, List, NamedTuple

import networkx as nx
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from rgnet.encoding.base_encoder import GraphEncoderBase, check_encoded_by_this
from rgnet.encoding.node_factory import Node, NodeFactory
from xmimir import (
    Atom,
    Literal,
    Predicate,
    XAtom,
    XDomain,
    XLiteral,
    XPredicate,
    XProblem,
    XState,
)


class PredicateEdgeType(NamedTuple):
    src_type: str
    pos: str
    dst_type: str


class HeteroGraphEncoder(GraphEncoderBase):
    """
    An encoder to represent states as heterogeneous graphs with objects and predicates as vertices
    and edges (i, j) whenever a predicate p(..., i, j, ...) holds in the state.

    """

    def __init__(
        self,
        domain: XDomain,
        node_factory: NodeFactory = NodeFactory(),
        obj_type_id: str = "obj",
    ) -> None:
        super().__init__(domain)
        self.obj_type_id: str = obj_type_id
        self.node_factory: NodeFactory = node_factory
        self.predicates: tuple[XPredicate, ...] = self.domain.predicates
        self.arity_dict: Dict[Node, int] = HeteroGraphEncoder.make_arity_dict(
            self.predicates, node_factory
        )
        # Generate all possible edge types
        self.all_edge_types: List[EdgeType] = []
        for predicate, arity in self.arity_dict.items():
            for pos in range(arity):
                self.all_edge_types.append((self.obj_type_id, str(pos), predicate))
                self.all_edge_types.append((predicate, str(pos), self.obj_type_id))

    def __eq__(self, other: HeteroGraphEncoder) -> bool:
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
                    (node_factory(predicate, **kwargs_dict), predicate.arity)
                    for predicate in predicates
                    for kwargs_dict in (
                        {"is_goal": True, "is_negated": False},
                        {"is_goal": False, "is_negated": False},
                        {"is_goal": True, "is_negated": True},
                    )
                ),
                key=operator.itemgetter(0),
            )
        )

    def encode(self, state: XState) -> nx.Graph:
        # Build hetero graph from state
        # One node for each object
        # One node for each atom
        # Edge label = position in atom
        problem = state.problem
        graph = nx.Graph(encoding=self, state=state)

        for obj in problem.objects:
            graph.add_node(self.node_factory(obj), type=self.obj_type_id)

        atom_or_literal: XAtom | XLiteral
        for atom_or_literal in self._atoms_and_goals_iterator(state):
            if isinstance(atom_or_literal, XLiteral):
                atom = atom_or_literal.atom
                is_goal = not hasattr(atom_or_literal, "is_not_goal")
            else:
                atom = atom_or_literal
                is_goal = False
            predicate = atom.predicate
            arity = predicate.arity
            if arity == 0:
                continue

            atom_node = self.node_factory(atom_or_literal)
            graph.add_node(
                atom_node, type=self.node_factory(predicate, is_goal=is_goal)
            )

            for pos, obj in enumerate(atom.objects):
                # Connect predicate node to object node
                graph.add_edge(self.node_factory(obj), atom_node, position=pos)
        return graph

    @check_encoded_by_this
    def to_pyg_data(self, graph: nx.Graph) -> HeteroData:
        del graph.graph["encoding"]
        del graph.graph["state"]
        nodes_dict: Dict[NodeType, List[Node]] = defaultdict(list)
        for node, node_type in nx.get_node_attributes(graph, "type").items():
            nodes_dict[node_type].append(node)

        node_idx_dict: Dict[NodeType, Dict[Node, int]] = {
            ntype: {node: i for i, node in enumerate(nodes)}
            for ntype, nodes in nodes_dict.items()
        }

        data = HeteroData()
        # Create x_dict, the feature matrix for each node type We don't have any
        # features for nodes, so we just create a zero tensor In order to be
        # independent of the model, we create a tensor of size 1 for object nodes and
        # a tensor of size arity for predicate nodes
        for node_type, nodes_of_type in nodes_dict.items():
            size = 1 if node_type == self.obj_type_id else self.arity_dict[node_type]
            data[node_type].x = torch.zeros(
                (len(nodes_of_type), size), dtype=torch.float32
            )

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

    def from_pyg_data(self, data: HeteroData) -> nx.Graph:
        """
        Reconstruct a graph from a HeteroData object.
        Every node has a type attribute, either self.obj_type_id or a predicate name.
        Node names are the concatenation of the type and the index in the feature matrix.
        Edge labels are the position in the atom.
        The returned graph is not the exact same as one returned by encode, but isomorphic.
        :param data: HetereData object encoded with this encoder.
        :return: The networkx graph as described above.
        """
        graph = nx.Graph(encoding=self)
        assert all(pred in data.node_types for pred in self.arity_dict.keys())
        obj_type: str = self.obj_type_id
        # Every node needs a unique name, but obj-names are lost, therefore we use the
        # index in the feature matrix together with the type.
        for obj_idx in range(data.x_dict[obj_type].shape[0]):
            graph.add_node(obj_type + str(obj_idx), type=obj_type)

        for predicate in self.arity_dict.keys():
            for pred_idx in range(data.x_dict[predicate].shape[0]):
                graph.add_node(predicate + str(pred_idx), type=predicate)

        for src, rel, dst in data.edge_types:
            src_tensor, dst_tensor = data.edge_index_dict[src, rel, dst]
            for src_idx, dst_idx in zip(src_tensor, dst_tensor):
                graph.add_edge(
                    src + str(src_idx.item()),
                    dst + str(dst_idx.item()),
                    position=int(rel),
                )
        return graph
