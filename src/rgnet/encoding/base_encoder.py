from __future__ import annotations

import functools
from abc import ABC, abstractmethod

import networkx as nx
import torch_geometric as pyg
from pymimir import State


class StateEncoderBase(ABC):
    """
    The state-graph encoder base class into an associated state-graph.
    """

    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def encode(self, state: State) -> nx.Graph | nx.DiGraph:
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
    def to_pyg_data(self, encoded_graph: nx.DiGraph) -> pyg.data.Data:
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


def check_encoded_by_this(func):
    @functools.wraps(func)
    def wrapper(self, graph, *args, **kwargs):
        if not self._encoded_by_this(graph):
            raise ValueError("Graph must have been encoded by this encoder")
        return func(self, graph, *args, **kwargs)

    return wrapper
