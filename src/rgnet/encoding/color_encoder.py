from __future__ import annotations

import itertools
import warnings
from collections import namedtuple
from copy import copy
from enum import Enum
from functools import singledispatchmethod
from types import NoneType
from typing import Any, Dict, Iterator, Optional, Tuple, TypeVar

import networkx as nx
import numpy as np
import torch_geometric as pyg
from pymimir import Atom, Domain, Literal, Object, Predicate, Problem, State, Type
from torch_geometric.data import Data

from rgnet.encoding.encoder_base import StateGraphEncoderBase
from rgnet.encoding.node_names import node_of

ColorKey = namedtuple("ColorKey", ["name", "position", "is_goal", "is_negated"])


class FeatureMode(Enum):
    # mere integer naming the category
    categorical = 0
    # one-hot vector encoding
    one_hot = 1
    # combinatorial encoding of features in a vector of user-determined length
    combinatorial = 2


class ColorGraphEncoder(StateGraphEncoderBase):
    """
    A state encoder into an associated colored state-graph for a specified domain.

    Each object will receive its own node, each predicate might receive its own node (if configured to do so),
    each atom will receive multiple nodes which
    """

    def __init__(
        self,
        domain: Domain,
        feature_mode: FeatureMode = FeatureMode.categorical,
        add_global_predicate_nodes: bool = False,
        feature_enc_len: Optional[int] = None,
    ):
        """
        Initialize the color graph encoder

        Parameters
        ----------
        domain: pymimir.Domain, the domain over which instance-states will be encoded
        add_global_predicate_nodes: bool, whether to add summarising predicate nodes to the graph.
            Predicate nodes will connect with respective pos-0-atom nodes, if applicable.
        """
        if not isinstance(feature_mode, FeatureMode):
            raise ValueError(
                f"`feature_mode` value {feature_mode} not element of the enum."
            )

        self.add_predicate_nodes = add_global_predicate_nodes
        self._domain = domain
        self._predicates = self.domain.predicates
        self._feature_mode = feature_mode
        self._feature_enc_len = feature_enc_len
        self._feature_lookup = self._build_feature_map(feature_mode, feature_enc_len)

    def __eq__(self, other: ColorGraphEncoder):
        return (
            self.add_predicate_nodes == other.add_predicate_nodes
            and self._domain == other.domain
            and self._feature_mode == other.feature_mode
            and self._feature_enc_len == other.feature_encoding_len
        )

    def _key_gen(self) -> Iterator[ColorKey]:
        for i, typ in enumerate(self._domain.types):
            yield ColorKey(typ.name, -1, False, False)
        for pred in self._predicates:
            for pos in range(0, max(1, pred.arity)):
                pos_or_none = pos if pred.arity > 0 else None
                for is_goal, is_negated in (
                    (False, False),
                    (True, False),
                    (True, True),
                ):
                    yield ColorKey(pred.name, pos_or_none, is_goal, is_negated)

    def _build_feature_map(self, mode: FeatureMode, encoding_len: Optional[int] = None):
        if mode == FeatureMode.categorical:
            none_feature = 0
            feature_iter = itertools.count(start=1)
        elif mode == FeatureMode.one_hot:
            encoding_len = sum(3 * max(1, pred.arity) for pred in self._predicates)
            none_feature = np.zeros(encoding_len, dtype=np.int8)
            feature_iter = np.eye(encoding_len, dtype=np.int8)
            # make read-only views
            none_feature.flags.writeable = False
            feature_iter.flags.writeable = False
        else:
            required_nr_states = sum(
                3 * max(1, pred.arity) for pred in self._predicates
            )
            if encoding_len is None:
                encoding_len, _ = divmod(required_nr_states, 2)
            else:
                # each position is a 0 or 1, encoding_len many positions -> 2^enc_len many different states possible,
                # only (0,0,...,0,0) is reserved as the null encoding (hence, -1)
                max_encoded_values = 2**encoding_len - 1
                if max_encoded_values < required_nr_states:
                    raise ValueError(
                        f"Given {encoding_len=} cannot represent all the necessary "
                        f"encoding states ({required_nr_states})"
                    )
                if encoding_len >= required_nr_states:
                    warnings.warn(
                        f"{encoding_len=} is no less than {required_nr_states=}. Will revert to one-hot encoding."
                    )
                    return self._build_feature_map(FeatureMode.one_hot)

            def feature_vector_gen():
                unit_matrix = np.eye(encoding_len, dtype=np.int8)
                # make read-only views
                unit_matrix.flags.writeable = False
                for idx_comb in itertools.chain(
                    itertools.combinations(range(encoding_len), n_combs)
                    for n_combs in range(1, encoding_len + 1)
                ):
                    # sum up the unit vectors of the given indices to create another feature vector
                    yield sum(unit_matrix[i] for i in idx_comb)

            none_feature = np.zeros(encoding_len, dtype=np.int8)
            # make read-only views
            none_feature.flags.writeable = False
            feature_iter = feature_vector_gen()

        key_iter = self._key_gen()
        colormap: Dict[Optional[ColorKey], np.ndarray] = dict(
            zip(key_iter, feature_iter)
        )
        colormap[None] = none_feature

        return colormap

    @property
    def domain(self):
        return self._domain

    @property
    def feature_mode(self):
        return self._feature_mode

    @property
    def feature_encoding_len(self):
        return self._feature_enc_len

    @property
    def feature_lookup(self):
        return copy(self._feature_lookup)

    def encode(self, state: State):
        problem = state.get_problem()
        graph = nx.Graph(encoding=self, state=state)

        for obj in problem.objects:
            graph.add_node(
                node_of(obj),
                feature=self.feature(obj.type),
                info=obj.type.name,
            )

        for atom_or_literal in itertools.chain(state.get_atoms(), problem.goal):
            if self.add_predicate_nodes:
                graph.add_node(
                    node_of(atom_or_literal, as_predicate=True),
                    feature=self.feature(None),
                )

            prev_predicate_node = None
            # only a literal has a member `atom`
            atom: Atom = getattr(atom_or_literal, "atom", atom_or_literal)
            if atom.predicate.arity == 0:
                graph.add_node(
                    node_of(atom_or_literal),
                    feature=self.feature(atom_or_literal),
                )
            for pos, obj in enumerate(atom.terms):
                object_node = node_of(obj, pos)
                atom_or_literal_node = node_of(atom_or_literal, pos)

                graph.add_node(
                    atom_or_literal_node,
                    feature=self.feature(atom_or_literal, pos),
                )
                # Connect atom node to object node
                graph.add_edge(object_node, atom_or_literal_node)
                if pos > 0:
                    # connect preceding positional node with the current one
                    graph.add_edge(prev_predicate_node, atom_or_literal_node)
                elif self.add_predicate_nodes:
                    # pos 0-node gets the connection to the predicate summarising node
                    graph.add_edge(
                        node_of(
                            atom_or_literal,
                            as_predicate=True,
                        ),
                        atom_or_literal_node,
                    )

                prev_predicate_node = atom_or_literal_node
        return graph

    def to_pyg_data(self, color_encoded_graph: nx.Graph) -> Data:
        # In the pyg.utils.from_networkx the graph is converted to a DiGraph
        # In this process it has to be pickled, which is not defined for pymimir objects
        del color_encoded_graph.graph["state"]
        del color_encoded_graph.graph["encoding"]
        # Every node has to have the same features
        for node, attr in color_encoded_graph.nodes.data():
            if "info" in attr:
                del attr["info"]
        data: Data = pyg.utils.from_networkx(color_encoded_graph)
        # We want floating point features
        data.x = data["feature"].float()
        # Ensure that every node has a vector of features even though its size is one
        data.x = data.x.view(-1, 1)
        return data

    @singledispatchmethod
    def feature(self, _: Any):
        return self._feature_lookup[None]

    @feature.register
    def _(self, atom: Atom, pos: int | None = None):
        if pos is None and atom.predicate.arity > 0:
            raise ValueError(
                f"atom {atom.get_name()} has arity {atom.predicate.arity} > 0, but given pos is None"
            )
        return self._feature_lookup[ColorKey(atom.predicate.name, pos, False, False)]

    @feature.register
    def _(self, literal: Literal, pos: int | None = None):
        atom = literal.atom
        if pos is None and atom.predicate.arity > 0:
            raise ValueError(
                f"atom {atom.get_name()} has arity {atom.predicate.arity} > 0, but given pos is None"
            )
        return self._feature_lookup[
            ColorKey(atom.predicate.name, pos, True, literal.negated)
        ]

    @feature.register
    def _(self, type_: Type, _: Any = None):
        return self._feature_lookup[ColorKey(type_.name, -1, False, False)]
