from __future__ import annotations

import itertools
import warnings
from collections import namedtuple
from copy import copy
from enum import Enum
from functools import cached_property, singledispatchmethod
from typing import Any, Dict, Iterator, Optional

import networkx as nx
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data

from rgnet.encoding.base_encoder import GraphEncoderBase, check_encoded_by_this
from rgnet.encoding.node_factory import NodeFactory
from xmimir import (
    Atom,
    GroundAtom,
    GroundLiteral,
    Literal,
    Predicate,
    XDomain,
    XLiteral,
    XState,
)

from .featuremap import FeatureMap


class ColorGraphEncoder(GraphEncoderBase):
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
        self._feature_map = feature_map or FeatureMap(domain)

    def __eq__(self, other: ColorGraphEncoder):
        return (
            self.predicate_nodes_enabled == other.predicate_nodes_enabled
            and self.domain == other.domain
            and self.feature_map == other.feature_map
        )

    @property
    def feature_map(self):
        return self._feature_map

    def encode(self, state: XState) -> nx.Graph:
        graph = nx.Graph(encoding=self, state=state)

        for obj in state.problem.objects:
            graph.add_node(self.node_factory(obj), feature=self.feature_map(None))

        atom_or_literal: XLiteral | Atom
        for atom_or_literal in self._atoms_and_goals_iterator(state):
            prev_predicate_node = None
            if isinstance(atom_or_literal, XLiteral):
                atom = atom_or_literal.atom
                is_goal = not hasattr(atom_or_literal.base, "is_not_goal")
            else:
                atom = atom_or_literal
                is_goal = False

            predicate = atom.predicate
            arity = predicate.arity
            if self.predicate_nodes_enabled:
                graph.add_node(
                    # global predicate nodes are never negated.
                    # The negation is encoded in the init features of the individual literals using this predicate
                    self.node_factory(predicate, is_goal=is_goal, is_negated=False),
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
        return graph

    @check_encoded_by_this
    def to_pyg_data(self, graph: nx.Graph) -> Data:
        # In the pyg.utils.from_networkx the graph is converted to a DiGraph
        # In this process it has to be pickled, which is not defined for pymimir objects
        del graph.graph["state"]
        del graph.graph["encoding"]
        data: Data = pyg.utils.from_networkx(graph, group_node_attrs=["feature"])
        return data
