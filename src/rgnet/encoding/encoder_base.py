from abc import ABC, abstractmethod

import networkx as nx
import torch_geometric as pyg
from pymimir import State


class StateEncoderBase(ABC):
    """
    The state-graph encoder base class into an associated state-graph.
    """

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
