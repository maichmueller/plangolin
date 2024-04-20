import itertools
from collections import namedtuple
from functools import singledispatchmethod
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch_geometric as pyg
from pymimir import Atom, Domain, Literal, Problem, State
from torch_geometric.data import Data

from rgnet.encoding.encoder_base import StateEncoderBase
from rgnet.encoding.node_names import node_of


class DirectStateEncoder(StateEncoderBase):
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

    def __init__(self, domain: Domain):
        """
        Initialize the direct graph encoder

        Parameters
        ----------
        domain: pymimir.Domain, the domain over which instance-states will be encoded
        """
        self._domain = domain
        self._feature_map, self._predicate_list = self._build_feature_map()

    def _build_feature_map(self):
        # times 3 because of augmentation for pred_name * (is normal atom / is goal atom / is negated goal atom)
        predicate_list = self.domain.predicates
        feature_dim = len(predicate_list) * 3
        # one-hot encoding of the (possibly (negated) goal) predicates
        edge_feature_map: np.ndarray = np.eye(feature_dim).reshape(
            len(predicate_list), 3, -1
        )
        return edge_feature_map, predicate_list

    @singledispatchmethod
    def feature_vector(self, item) -> np.ndarray:
        raise NotImplementedError("Type passed not supported: {type(item)}")

    @feature_vector.register
    def atom_feature_vector(self, atom: Atom) -> np.ndarray:
        return self._feature_map[self._predicate_list.index(atom.predicate), 0]

    @feature_vector.register
    def literal_feature_vector(self, literal: Literal) -> np.ndarray:
        return self._feature_map[
            self._predicate_list.index(literal.atom.predicate), 1 + literal.negated
        ]

    @property
    def domain(self):
        return self._domain

    @staticmethod
    def _add_feature_vector(graph: nx.DiGraph, from_: str, to: str, vector: np.ndarray):
        if graph.has_edge(from_, to):
            graph.edges[from_, to]["feature"] += vector
        else:
            graph.add_edge(from_, to, feature=vector)

    def encode(self, state: State):
        problem = state.get_problem()
        graph = nx.DiGraph(encoding=self, state=state)

        for obj in problem.objects:
            graph.add_node(
                node_of(obj),
                info=obj.type.name,
            )

        for atom_or_literal in itertools.chain(state.get_atoms(), problem.goal):
            # only a literal has a member `atom`
            atom: Atom = getattr(atom_or_literal, "atom", atom_or_literal)
            if atom.predicate.arity == 0:
                # TODO: what to do with arity-0 atoms? do these become extra nodes?
                ...
            else:
                objs = atom.terms
                for i, obj in enumerate(objs):
                    next_obj = objs[i + 1] if len(objs) > 1 else obj
                    graph.add_edge(
                        node_of(obj),
                        node_of(next_obj),
                        feature=self.feature_vector(atom_or_literal),
                    )
            return graph

    def to_pyg_data(self, direct_encoded_graph: nx.DiGraph) -> Data:
        # In the pyg.utils.from_networkx the graph is converted to a DiGraph
        # In this process it has to be pickled, which is not defined for pymimir.State
        del direct_encoded_graph.graph["state"]
        # Every node has to have the same features
        for node, attr in direct_encoded_graph.nodes.data():
            if "info" in attr:
                del attr["info"]
        data: Data = pyg.utils.from_networkx(direct_encoded_graph)
        # We want floating point features
        data.x = data["feature"].float()
        # Ensure that every node has a vector of features even though its size is one
        data.x = data.x.view(-1, 1)
        return data
