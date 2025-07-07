from __future__ import annotations

import logging
import operator
from collections import defaultdict
from functools import cache
from typing import Dict, List, NamedTuple, Sequence

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


class HeteroGraphEncoder(GraphEncoderBase[nx.MultiGraph]):
    """
    An encoder to represent states as heterogeneous graphs with objects and predicates as vertices
    and edges (i, j) whenever a predicate p(..., i, j, ...) holds in the state.

    """

    goal_satisfied_suffix = "_sat"

    def __init__(
        self,
        domain: XDomain,
        node_factory: NodeFactory | None = None,
        obj_type_id: str = "obj",
        add_goal_satisfied_atoms: bool = False,
    ) -> None:
        super().__init__(domain)
        self.obj_type_id: str = obj_type_id
        # Initialize the default here to please jsonargparse.
        self.node_factory: NodeFactory = node_factory or NodeFactory()
        self.add_goal_satisfied_atoms = add_goal_satisfied_atoms
        self.predicates: tuple[XPredicate, ...] = self.domain.predicates()
        self.arity_dict: Dict[Node, int] = HeteroGraphEncoder.make_arity_dict(
            self.predicates,
            self.node_factory,
            add_goal_satisfied_atoms=add_goal_satisfied_atoms,
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
            and self.add_goal_satisfied_atoms == other.add_goal_satisfied_atoms
        )

    @staticmethod
    @cache
    def make_arity_dict(
        predicates: tuple[XPredicate],
        node_factory: NodeFactory = NodeFactory(),
        add_goal_satisfied_atoms: bool = False,
    ) -> Dict[Node, int]:
        ad = [
            (node_factory(predicate, **kwargs_dict), predicate.arity)
            for predicate in predicates
            for kwargs_dict in (
                {"is_goal": True, "is_negated": False},
                {"is_goal": False, "is_negated": False},
                {"is_goal": True, "is_negated": True},
            )
        ]
        if add_goal_satisfied_atoms:
            for predicate in predicates:
                ad.append(
                    (
                        f"{node_factory(predicate, is_goal=True, is_negated=False)}"
                        f"{HeteroGraphEncoder.goal_satisfied_suffix}",
                        predicate.arity,
                    )
                )
        ad = dict(sorted(ad, key=operator.itemgetter(0)))
        return ad

    def _encode(
        self,
        items: Sequence[XAtom | XLiteral],
        graph: GraphT,
    ) -> None:
        # Build hetero graph from state
        # One node for each object
        # One node for each atom
        # Edge label = position in atom

        for obj in self._contained_objects(items):
            graph.add_node(
                self.node_factory(obj), type=self.obj_type_id, name=obj.get_name()
            )

        atom_or_literal: XAtom | XLiteral
        for atom_or_literal in items:
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

            atom_or_literal_node = self.node_factory(atom_or_literal)
            graph.add_node(
                atom_or_literal_node, type=self.node_factory(predicate, is_goal=is_goal)
            )

            for pos, obj in enumerate(atom.objects):
                # Connect predicate node to object node
                graph.add_edge(
                    self.node_factory(obj), atom_or_literal_node, position=pos
                )
        # Add goal satisfied atoms if requested
        if self.add_goal_satisfied_atoms:
            goals = []
            actual_atoms = []
            for item in items:
                if isinstance(item, XAtom):
                    actual_atoms.append(item)
                else:
                    goals.append(item)
            satisfied_goals = [
                goal
                for goal in goals
                if any(goal.atom.semantic_eq(atom) for atom in actual_atoms)
                != goal.is_negated
            ]
            for goal in satisfied_goals:
                literal_node = self.node_factory(goal) + self.goal_satisfied_suffix
                graph.add_node(
                    literal_node,
                    type=self.node_factory(goal.atom.predicate, is_goal=True)
                    + self.goal_satisfied_suffix,
                )
                for pos, obj in enumerate(goal.atom.objects):
                    # Connect satisfied goal node to object node
                    graph.add_edge(self.node_factory(obj), literal_node, position=pos)

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
            get_logger(__name__).warning(f"No object in graph ({graph})")
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
        Every node has a type attribute, either self.obj_type_id or a ext_predicate name.
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
        if predicates - (
            self.arity_dict.keys()
            | {
                f"{pred}{self.node_factory.goal_suffix}"
                for pred in self.arity_dict.keys()
            }
            | {
                f"{self.node_factory.negation_prefix}{pred}{self.node_factory.goal_suffix}"
                for pred in self.arity_dict.keys()
            }
            | {
                f"{self.node_factory.negation_prefix}{pred}"
                for pred in self.arity_dict.keys()
            }
            | {
                f"{pred}{self.node_factory.goal_suffix}{self.goal_satisfied_suffix}"
                for pred in self.arity_dict.keys()
            }
            | {
                f"{self.node_factory.negation_prefix}{pred}{self.node_factory.goal_suffix}{self.goal_satisfied_suffix}"
                for pred in self.arity_dict.keys()
            }
        ):
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
                if src != self.obj_type_id:
                    src_idx = src_idx.item()
                    atoms[(src, src_idx)][int(rel)] = obj_names[dst_idx.item()]
                elif dst != self.obj_type_id:
                    dst_idx = dst_idx.item()
                    atoms[(dst, dst_idx)][int(rel)] = obj_names[src_idx.item()]
                else:
                    raise ValueError(
                        f"Both src and dst are objects: {edge_type=}, {src_idx=}, {dst_idx=}."
                    )
        for (ext_predicate, atom_idx), obj_dict in atoms.items():
            pos_obj_tuples = sorted(obj_dict.items(), key=operator.itemgetter(0))
            predicate = ext_predicate[:]
            prefix, goal_suffix, goal_sat_suffix = "", "", ""
            if predicate.endswith(self.goal_satisfied_suffix):
                predicate = predicate[: -len(self.goal_satisfied_suffix)]
                goal_sat_suffix = self.goal_satisfied_suffix
            if predicate.endswith(self.node_factory.goal_suffix):
                predicate = predicate[: -len(self.node_factory.goal_suffix)]
                goal_suffix = self.node_factory.goal_suffix
            if ext_predicate.startswith(self.node_factory.negation_prefix):
                predicate = predicate[len(self.node_factory.negation_prefix) :]
                prefix = self.node_factory.negation_prefix
            atom_str = atom_str_template.render(
                predicate=predicate,
                objects=[obj for pos, obj in pos_obj_tuples],
            )
            atom_str = f"{prefix}{atom_str}{goal_suffix}{goal_sat_suffix}"
            graph.add_node(atom_str, type=ext_predicate)
            for pos, obj in pos_obj_tuples:
                graph.add_edge(
                    atom_str,
                    obj,
                    position=int(pos),
                )
        return self._graph_t(graph)
