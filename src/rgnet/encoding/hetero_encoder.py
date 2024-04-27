from __future__ import annotations

import itertools
import logging
from collections import defaultdict, namedtuple
from typing import Dict, List

import networkx as nx
import torch
from pymimir import Atom, Domain, Literal, Predicate, State
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from rgnet.encoding.base_encoder import StateEncoderBase
from rgnet.encoding.node_factory import Node, NodeFactory

PredicateEdgeType = namedtuple("PredicateEdgeType", ["src_type", "pos", "dst_type"])


class HeteroGraphEncoder(StateEncoderBase):
    def __init__(
        self,
        domain: Domain,
        hidden_size: int,  # TODO Decouple hidden_size from encoding
        node_factory: NodeFactory = NodeFactory(),
        obj_type_id: str = "obj",
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.obj_type_id: str = obj_type_id
        self.node_factory: NodeFactory = node_factory
        self.predicates: List[Predicate] = domain.predicates
        self.arity_dict: Dict[Node, int] = dict(
            sorted(
                itertools.chain(
                    map(
                        lambda p: (
                            self.node_factory(p, is_goal=False, is_negated=False),
                            p.arity,
                        ),
                        self.predicates,
                    ),
                    map(
                        lambda p: (
                            self.node_factory(p, is_goal=True, is_negated=False),
                            p.arity,
                        ),
                        self.predicates,
                    ),
                    map(
                        lambda p: (
                            self.node_factory(p, is_goal=True, is_negated=True),
                            p.arity,
                        ),
                        self.predicates,
                    ),
                ),
                key=lambda x: x[0],
            )
        )
        # Generate all possible edge types
        self.all_edge_types: List[EdgeType] = []
        for predicate, arity in self.arity_dict.items():
            for pos in range(arity):
                self.all_edge_types.append((self.obj_type_id, str(pos), predicate))
                self.all_edge_types.append((predicate, str(pos), self.obj_type_id))

    def __eq__(self, other: HeteroGraphEncoder) -> bool:
        return (
            self.hidden_size == other.hidden_size
            and self.obj_type_id == other.obj_type_id
            and self.predicates == other.predicates
        )

    def encode(self, state: State) -> nx.Graph:
        # Build hetero graph from state
        # One node for each object
        # One node for each atom
        # Edge label = position in atom
        problem = state.get_problem()
        graph = nx.Graph(encoding=self, state=state)

        for obj in problem.objects:
            graph.add_node(self.node_factory(obj), type=self.obj_type_id)

        atom_or_literal: Atom | Literal
        for atom_or_literal in itertools.chain(state.get_atoms(), problem.goal):
            atom: Atom = getattr(atom_or_literal, "atom", atom_or_literal)
            if atom.predicate.arity == 0:
                continue

            atom_node = self.node_factory(atom_or_literal)
            graph.add_node(
                atom_node, type=self.node_factory(atom_or_literal, as_predicate=True)
            )

            for pos, obj in enumerate(atom.terms):
                # Connect predicate node to object node
                graph.add_edge(self.node_factory(obj), atom_node, position=pos)
        return graph

    def to_pyg_data(self, graph: nx.Graph) -> HeteroData:
        if not self._encoded_by_this(graph):
            raise ValueError("Graph must have been encoded by this encoder")
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
        # Create x_dict, the feature matrix for each node type
        for node_type, nodes_of_type in nodes_dict.items():
            hidden = (
                self.hidden_size
                if node_type == self.obj_type_id
                else self.hidden_size * self.arity_dict[node_type]
            )
            data[node_type].x = torch.zeros(
                (len(nodes_of_type), hidden), dtype=torch.float32
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
