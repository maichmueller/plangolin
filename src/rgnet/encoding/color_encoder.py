import itertools
from collections import namedtuple
from functools import singledispatchmethod
from types import NoneType
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import torch_geometric as pyg
from pymimir import Atom, Domain, Literal, Object, Predicate, Problem, State, Type
from torch_geometric.data import Data

from rgnet.encoding.encoder_base import StateEncoderBase

ColorKey = namedtuple("ColorKey", ["name", "position", "is_goal", "is_negated"])


class ColorGraphEncoder(StateEncoderBase):
    """
    A state encoder into an associated colored state-graph for a specified domain.

    Each object will receive its own node, each predicate might receive its own node (if configured to do so),
    each atom will receive multiple nodes which
    """

    def __init__(self, domain: Domain, add_global_predicate_nodes: bool = False):
        """
        Initialize the color graph encoder

        Parameters
        ----------
        domain: pymimir.Domain, the domain over which instance-states will be encoded
        add_global_predicate_nodes: bool, whether to add summarising predicate nodes to the graph.
            Predicate nodes will connect with respective pos-0-atom nodes, if applicable.
        """
        self._domain = domain
        self._feature_map = self._build_feature_map()
        self.add_predicate_nodes = add_global_predicate_nodes

    def _build_feature_map(self):
        colormap: Dict[ColorKey | None, int] = {None: 0}
        feature = 1
        for i, typ in enumerate(self.domain.types):
            colormap[ColorKey(typ.name, -1, False, False)] = feature
            feature += 1
        for pred in self.domain.predicates:
            for pos in range(0, max(1, pred.arity)):
                pos_or_none = pos if pred.arity > 0 else None
                colormap[
                    ColorKey(pred.name, pos_or_none, is_goal=False, is_negated=False)
                ] = feature
                colormap[
                    ColorKey(pred.name, pos_or_none, is_goal=True, is_negated=False)
                ] = (feature + 1)
                colormap[
                    ColorKey(pred.name, pos_or_none, is_goal=True, is_negated=True)
                ] = (feature + 2)
                feature += 3
        return colormap

    @property
    def domain(self):
        return self._domain

    @property
    def feature_mapping(self):
        return self._feature_map

    def encode(self, state: State):
        problem = state.get_problem()
        graph = nx.Graph(encoding=self, state=state)

        for obj in problem.objects:
            graph.add_node(
                self.node_of(obj),
                feature=self.feature(obj.type),
                info=obj.type.name,
            )

        for atom_or_literal in itertools.chain(state.get_atoms(), problem.goal):
            if self.add_predicate_nodes:
                graph.add_node(
                    self.node_of(atom_or_literal, as_predicate=True),
                    feature=self.feature(None),
                )

            prev_predicate_node = None
            # only a literal has a member `atom`
            atom: Atom = getattr(atom_or_literal, "atom", atom_or_literal)
            if atom.predicate.arity == 0:
                graph.add_node(
                    self.node_of(atom_or_literal),
                    feature=self.feature(atom_or_literal),
                )
            for pos, obj in enumerate(atom.terms):
                object_node = self.node_of(obj, pos)
                atom_or_literal_node = self.node_of(atom_or_literal, pos)

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
                        self.node_of(
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
        return self._feature_map[None]

    @feature.register
    def _(self, atom: Atom, pos: int | None = None):
        if pos is None and atom.predicate.arity > 0:
            raise ValueError(
                f"atom {atom.get_name()} has arity {atom.predicate.arity} > 0, but given pos is None"
            )
        return self._feature_map[ColorKey(atom.predicate.name, pos, False, False)]

    @feature.register
    def _(self, literal: Literal, pos: int | None = None):
        atom = literal.atom
        if pos is None and atom.predicate.arity > 0:
            raise ValueError(
                f"atom {atom.get_name()} has arity {atom.predicate.arity} > 0, but given pos is None"
            )
        return self._feature_map[
            ColorKey(atom.predicate.name, pos, True, literal.negated)
        ]

    @feature.register
    def _(self, type_: Type, _: Any = None):
        return self._feature_map[ColorKey(type_.name, -1, False, False)]

    @singledispatchmethod
    def node_of(self, *args, **kwargs) -> str | None:
        return None

    @node_of.register
    def atom_node(
        self,
        atom: Atom,
        pos: int | None = None,
        *args,
        as_predicate: bool = False,
        **kwargs,
    ) -> str | None:
        if as_predicate:
            return self.node_of(atom.predicate, is_goal=False, is_negated=False)
        return f"{atom.get_name()}:{pos}"

    @node_of.register
    def predicate_node(
        self, predicate: Predicate, *, is_goal: bool, is_negated: bool, **kwargs
    ) -> str | None:
        if not self.add_predicate_nodes:
            return None
        negation = "~" if is_negated else ""
        suffix = "_g" if is_goal else ""
        return f"{negation}{predicate.name}{suffix}"

    @node_of.register
    def literal_node(
        self,
        literal: Literal,
        pos: int | None = None,
        *,
        as_predicate: bool = False,
        **kwargs,
    ) -> str | None:
        if as_predicate:
            return self.node_of(
                literal.atom.predicate, is_goal=True, is_negated=literal.negated
            )
        negation = "~" if literal.negated else ""
        pos_string = f":{pos}" if pos is not None else ""
        return f"{negation}{literal.atom.get_name()}_g{pos_string}"

    @node_of.register
    def object_node(self, obj: Object, *args, **kwargs) -> str | None:
        return obj.name
