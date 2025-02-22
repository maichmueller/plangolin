from __future__ import annotations

import itertools
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data

from xmimir import Atom, XAtom, XDomain, XLiteral

from .base_encoder import GraphEncoderBase, GraphT, check_encoded_by_this
from .featuremap import FeatureMap, FeatureMode
from .node_factory import Node, NodeFactory


class DirectGraphEncoder(GraphEncoderBase[nx.DiGraph]):
    """
    An encoder to represent states as directed graphs with objects as vertices
    and edges (i, j) whenever a predicate p(..., i, j, ...) holds in the state.

    Which predicate creates the edge between `i` and `j` is encoded as the sum
    of one-hot encodings of all predicates (including goal predicates and
    negated goal predicates). For example, an edge (i, k) would have the
    feature vector E = (1, 0, 1, 0, 0) over the fixed predicates mapping
    (p1, p2, p1_GOAL, p2_GOAL, NOT_p1_GOAL, NOT_p2_GOAL) to represent active
    predicates p1(..., i, j, ...) and p1_GOAL(..., p, j, ...) in the state `s`.
    """

    class aux_node:
        def __repr__(self):
            return "__aux__"

    _auxiliary_node = aux_node()

    def __init__(
        self,
        domain: XDomain,
        node_factory: NodeFactory = NodeFactory(),
        feature_map: Optional[FeatureMap] = None,
    ):
        """
        Initialize the direct graph encoder

        Parameters
        ----------
        domain: pymimir.Domain, the domain over which instance-states will be encoded
        """

        # register our aux node to the node factory's dispatch
        @node_factory.__call__.register
        def _(self_, aux: DirectGraphEncoder.aux_node, *args, **kwargs):
            return str(aux)

        super().__init__(domain)
        self.node_factory = node_factory
        self._feature_map = feature_map or FeatureMap(
            domain,
            mode=FeatureMode.one_hot,
            ignore_arg_position=True,
        )
        assert (
            self._feature_map.mode == FeatureMode.one_hot
            and self._feature_map.ignore_arg_position
        ), "Only one-hot encoding without arg-positions supported."

    def __eq__(self, other: DirectGraphEncoder):
        return self.domain == other.domain and self._feature_map == other._feature_map

    @staticmethod
    def _emplace_feature(
        graph: nx.DiGraph, source_obj: Node, target_obj: Node, feature: np.ndarray
    ):
        if graph.has_edge(source_obj, target_obj):
            # in-place add to combine 1s of the predicate positions that
            # are enabling communication between the objects
            curr_feature = graph.edges[source_obj, target_obj]["feature"]
            graph.edges[source_obj, target_obj]["feature"] = curr_feature + feature
        else:
            graph.add_edge(source_obj, target_obj, feature=feature)

    def _encode(self, items: Sequence[XAtom] | Sequence[XLiteral], graph: GraphT):
        objects = self._contained_objects(items)
        for obj in objects:
            graph.add_node(self.node_factory(obj), feature=self._feature_map(None))
        graph.add_node(
            self.node_factory(self._auxiliary_node), feature=self._feature_map(None)
        )

        for atom_or_literal in items:
            # only a literal has a member `get_atom`
            atom: Atom = getattr(atom_or_literal, "atom", atom_or_literal)
            is_goal = not hasattr(atom_or_literal, "is_not_goal")
            predicate = atom.predicate
            arity = predicate.arity
            if arity == 0:
                # for 0 arity atoms, the aux nodes sends its regards to all objects
                source_objs = itertools.repeat(self._auxiliary_node)
                target_objs = objects
            elif arity == 1:
                # a 1 arity atom creates only self-edges
                source_objs = atom.objects
                target_objs = source_objs
            else:
                # a 2+ arity atom creates edge chains between successor objects as ordered by the terms list
                source_objs = atom.objects
                target_objs = source_objs[1:]

            feature = self._feature_map(atom_or_literal, is_goal=is_goal)
            for src_obj, tgt_obj in zip(source_objs, target_objs):
                self._emplace_feature(
                    graph,
                    self.node_factory(src_obj),
                    self.node_factory(tgt_obj),
                    feature,
                )
        return graph

    @check_encoded_by_this
    def to_pyg_data(self, graph: GraphT) -> Data:
        # In the pyg.utils.from_networkx the graph is converted to a DiGraph
        # In this process it has to be pickled, which is not defined for pymimir.State
        del graph.graph["state"]
        del graph.graph["encoding"]
        # Every node has to have the same features
        for node, attr in graph.nodes.data():
            if "info" in attr:
                del attr["info"]
        data: Data = pyg.utils.from_networkx(graph)
        # We want floating point features
        data.x = data["feature"].float()
        data.edge_attr = data.edge_attr.float()
        # Ensure that every node has a vector of features even though its size is one
        data.x = data.x.view(-1, 1)
        return data
