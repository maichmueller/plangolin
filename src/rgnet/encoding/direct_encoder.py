from __future__ import annotations

import itertools
from functools import singledispatchmethod
from types import NoneType

import networkx as nx
import numpy as np
import torch_geometric as pyg
from pymimir import Atom, Domain, Literal, State
from torch_geometric.data import Data

from rgnet.encoding.base_encoder import StateEncoderBase, check_encoded_by_this
from rgnet.encoding.node_factory import Node, NodeFactory


class DirectGraphEncoder(StateEncoderBase):
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
        domain: Domain,
        node_factory: NodeFactory = NodeFactory(),
    ):
        """
        Initialize the direct graph encoder

        Parameters
        ----------
        domain: pymimir.Domain, the domain over which instance-states will be encoded
        """

        # register our aux node to the node factory's dispatch
        @node_factory.__call__.register
        def _(self2, aux: DirectGraphEncoder.aux_node, *args, **kwargs):
            return str(aux)

        self.node_factory = node_factory
        self._domain = domain
        self._predicates = self.domain.predicates
        self._feature_map = self._build_feature_map()

    def _build_feature_map(self):
        # times 3 because of augmentation for pred_name * (is normal atom / is goal atom / is negated goal atom)
        feature_dim = len(self._predicates) * 3
        # one-hot encoding of the (possibly (negated) goal) predicates
        edge_feature_vectors: np.ndarray = np.eye(feature_dim, dtype=np.int8).reshape(
            len(self._predicates), 3, feature_dim
        )
        # make all views read only
        edge_feature_vectors.flags.writeable = False
        return edge_feature_vectors

    def __eq__(self, other: DirectGraphEncoder):
        return self._domain == other.domain

    @singledispatchmethod
    def feature(self, item) -> np.ndarray:
        raise NotImplementedError(f"Type passed not supported: {type(item)}")

    @feature.register
    def atom_feature_vector(self, atom: Atom) -> np.ndarray:
        return self._feature_map[self._predicates.index(atom.predicate), 0]

    @feature.register
    def literal_feature_vector(self, literal: Literal) -> np.ndarray:
        return self._feature_map[
            self._predicates.index(literal.atom.predicate), 1 + literal.negated
        ]

    @feature.register(NoneType)
    def none_feature_vector(self, _) -> np.ndarray:
        return np.zeros(self._feature_map.shape[0], dtype=np.int8)

    @property
    def domain(self):
        return self._domain

    @staticmethod
    def _emplace_feature(
        graph: nx.DiGraph, source_obj: Node, target_obj: Node, feature: np.ndarray
    ):
        if graph.has_edge(source_obj, target_obj):
            # in-place add to combine 1s of  the predicate positions that
            # are enabling communication between the objects
            curr_feature = graph.edges[source_obj, target_obj]["feature"]
            graph.edges[source_obj, target_obj]["feature"] = curr_feature + feature
        else:
            graph.add_edge(source_obj, target_obj, feature=feature)

    def encode(self, state: State):
        problem = state.get_problem()
        graph = nx.DiGraph(encoding=self, state=state)

        objects = problem.objects
        for obj in objects:
            graph.add_node(
                self.node_factory(obj),
                feature=self.feature(None),
                info=obj.type.name,
            )
        graph.add_node(
            self.node_factory(self._auxiliary_node), feature=self.feature(None)
        )

        for atom_or_literal in itertools.chain(state.get_atoms(), problem.goal):
            # only a literal has a member `atom`
            atom: Atom = getattr(atom_or_literal, "atom", atom_or_literal)
            arity = atom.predicate.arity
            if arity == 0:
                # for 0 arity atoms, the aux nodes sends its regards to all objects
                source_objs = itertools.repeat(self._auxiliary_node)
                target_objs = objects
            elif arity == 1:
                # a 1 arity atom creates only self-edges
                source_objs = atom.terms
                target_objs = source_objs
            else:
                # a 2+ arity atom creates edge chains between successor objects as ordered by the terms list
                source_objs = atom.terms
                target_objs = source_objs[1:]

            for src_obj, tgt_obj in zip(source_objs, target_objs):
                self._emplace_feature(
                    graph,
                    self.node_factory(src_obj),
                    self.node_factory(tgt_obj),
                    self.feature(atom_or_literal),
                )
        return graph

    @check_encoded_by_this
    def to_pyg_data(self, graph: nx.DiGraph) -> Data:
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
