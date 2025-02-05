from __future__ import annotations

import functools
import itertools
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Type

import networkx as nx
import torch_geometric as pyg

from xmimir import (
    Domain,
    PDDLRepositories,
    Problem,
    State,
    StaticLiteral,
    XAtom,
    XCategory,
    XDomain,
    XLiteral,
    XProblem,
    XState,
)


def _patch_as_nongoal(literal: StaticLiteral):
    # monkey patch the literal to be a non-goal (only for internal use)
    literal.is_not_goal = True
    return literal


class GraphEncoderBase(ABC):
    """
    The state-graph encoder base class into an associated state-graph.
    """

    def __init__(
        self,
        domain: XDomain,
        *args,
        **kwargs,
    ):
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def encode(self, state: XState) -> nx.Graph | nx.DiGraph:
        """
        Encodes the state into a networkx-graph representation.

        Parameters
        ----------
        state: pymimir.State, the state to encode as a graph.

        Returns
        -------
        nx.Graph or nx.DiGraph
        """
        ...

    @abstractmethod
    def to_pyg_data(
        self, encoded_graph: nx.DiGraph
    ) -> pyg.data.Data | pyg.data.HeteroData:
        """
        Converts the encoded state into a torch-geometric data object.
        Parameters
        ----------
        encoded_graph: nx.DiGraph or nx.Graph, the graph to convert.

        Returns
        -------
        pyg.data.Data
        """
        ...

    def _encoded_by_this(self, graph: nx.Graph | nx.DiGraph) -> bool:
        return (
            hasattr(graph, "graph")
            and "encoding" in graph.graph
            and graph.graph["encoding"] == self
        )

    def _atoms_and_goals_iterator(self, state: XState) -> Iterable[XLiteral | XAtom]:
        return itertools.chain(
            (
                XLiteral(_patch_as_nongoal(l))
                for l in state.problem.base.get_static_initial_literals()
            ),
            state.problem.goal(XCategory.fluent, XCategory.derived),
            state.atoms(),
        )


def check_encoded_by_this(func):
    @functools.wraps(func)
    def wrapper(self, graph, *args, **kwargs):
        if not self._encoded_by_this(graph):
            raise ValueError("Graph must have been encoded by this encoder")
        return func(self, graph, *args, **kwargs)

    return wrapper


class EncoderFactory:
    def __init__(
        self, encoder_class: Type[GraphEncoderBase], kwargs: Optional[dict] = None
    ):
        self.encoder_class = encoder_class
        self.kwargs = kwargs or dict()

    def __call__(self, domain: XDomain) -> GraphEncoderBase:
        return self.encoder_class(domain, **self.kwargs)
