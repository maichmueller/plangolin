from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx
import torch_geometric as pyg
from torch_geometric.data import Data

from xmimir import XAtom, XDomain, XLiteral

from .base_encoder import (
    EncoderFactory,
    GraphEncoderBase,
    GraphT,
    check_encoded_by_this,
)
from .featuremap import FeatureMap
from .node_factory import NodeFactory


class ColorGraphEncoder(GraphEncoderBase[nx.Graph]):
    """
    A state encoder into an associated colored state-graph for a specified domain.

    Each object will receive its own node, each predicate might receive its own node (if configured to do so),
    each atom will receive multiple nodes which
    """

    def __init__(
        self,
        domain: XDomain,
        feature_map: Optional[FeatureMap] = None,
        node_factory: NodeFactory = NodeFactory(),
        enable_global_predicate_nodes: bool = False,
    ):
        """
        Initialize the color graph encoder

        Parameters
        ----------
        domain: xmimir.XProblem, the problem over which instance-states will be encoded
        enable_global_predicate_nodes: bool, whether to add summarising predicate nodes to the graph.
            Predicate nodes will connect with respective pos-0-atom nodes, if applicable.
        """
        super().__init__(domain)
        self.node_factory = node_factory
        self.predicate_nodes_enabled = enable_global_predicate_nodes
        self._feature_map = feature_map or FeatureMap(domain.predicates())

    def __eq__(self, other: ColorGraphEncoder):
        return (
            self.predicate_nodes_enabled == other.predicate_nodes_enabled
            and self.domain == other.domain
            and self.feature_map == other.feature_map
        )

    def as_factory(self) -> EncoderFactory:
        return EncoderFactory(
            self.__class__,
            kwargs={
                "node_factory": self.node_factory,
                "feature_map": self._feature_map,
                "enable_global_predicate_nodes": self.predicate_nodes_enabled,
            },
        )

    @property
    def feature_map(self):
        return self._feature_map

    def _encode(self, items: Sequence[XAtom | XLiteral], graph: GraphT):
        for obj in self._contained_objects(items):
            graph.add_node(self.node_factory(obj), feature=self.feature_map(None))

        atom_or_literal: XLiteral | XAtom
        for atom_or_literal in items:
            prev_predicate_node = None
            if isinstance(atom_or_literal, XLiteral):
                is_goal = True
                atom = atom_or_literal.atom
            else:
                is_goal = False
                atom = atom_or_literal
            predicate = atom.predicate
            arity = predicate.arity
            if self.predicate_nodes_enabled:
                graph.add_node(
                    # global predicate nodes are never negated.
                    # The negation is encoded in the init features of the individual literals using this predicate
                    self.node_factory(
                        predicate,
                        is_goal=is_goal,
                        is_negated=False,
                    ),
                    feature=self.feature_map(None),
                )

            if arity == 0:
                graph.add_node(
                    self.node_factory(
                        atom_or_literal,
                        is_goal=is_goal,
                    ),
                    feature=self.feature_map(atom_or_literal, is_goal=is_goal),
                )
            for pos, obj in enumerate(atom.objects):
                object_node = self.node_factory(obj, pos)
                atom_or_literal_node = self.node_factory(
                    atom_or_literal,
                    pos,
                    is_goal=is_goal,
                )

                graph.add_node(
                    atom_or_literal_node,
                    feature=self.feature_map(atom_or_literal, pos, is_goal=is_goal),
                )
                # Connect atom node to object node
                graph.add_edge(object_node, atom_or_literal_node)
                if pos > 0:
                    # connect preceding positional node with the current one
                    graph.add_edge(prev_predicate_node, atom_or_literal_node)
                elif self.predicate_nodes_enabled:
                    # pos 0-node gets the connection to the predicate summarising node
                    graph.add_edge(
                        self.node_factory(predicate, is_goal=is_goal),
                        atom_or_literal_node,
                    )

                prev_predicate_node = atom_or_literal_node

    @check_encoded_by_this
    def to_pyg_data(self, graph: GraphT) -> Data:
        # In the pyg.utils.from_networkx the graph is converted to a DiGraph
        # In this process it has to be pickled, which is not defined for pymimir objects
        del graph.graph["state"]
        del graph.graph["encoding"]
        data: Data = pyg.utils.from_networkx(graph, group_node_attrs=["feature"])
        return data
