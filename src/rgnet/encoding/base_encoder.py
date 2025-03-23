from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Optional, Sequence, Type, TypeVar, get_args

import networkx as nx
import torch_geometric as pyg

from xmimir import Object, XAtom, XDomain, XLiteral, XState, gather_objects

GraphT = TypeVar("GraphT", nx.Graph, nx.DiGraph)


class GraphEncoderBase(ABC, Generic[GraphT]):
    """
    The state-graph encoder base class into an associated state-graph.
    """

    _graph_t: Any

    def __init__(
        self,
        domain: XDomain,
        *args,
        **kwargs,
    ):
        self._domain = domain

    def __init_subclass__(cls) -> None:
        # get the explicit graph type of the encoder
        try:
            cls._graph_t = get_args(cls.__orig_bases__[0])[0]
        except (AttributeError, IndexError) as e:
            raise NotImplementedError(
                f"Could not extract specific graph type from generic inheritance. "
                f"Remember to explicate the graph type when subclassing {GraphEncoderBase.__name__} as e.g. such "
                f"`{GraphEncoderBase.__name__}[nx.Graph].",
            ) from e

    @property
    def domain(self):
        return self._domain

    @abstractmethod
    def __eq__(self, other): ...

    def encode(
        self, state: XState | Iterable[XAtom] | Iterable[XLiteral], **kwargs
    ) -> GraphT:
        """
        Encodes the state/atoms/literals into a networkx-graph representation.

        Parameters
        ----------
        state: XState | Iterable[XAtom] | Iterable[XLiteral],
            the state to encode as a graph.

        Returns
        -------
        nx.Graph or nx.DiGraph
        """

        if isinstance(state, XState):
            graph = self._graph_t(encoding=self, state=state)
            self._encode(
                tuple(state.atoms(with_statics=True)),
                graph,
            )
            self._encode(tuple(state.problem.goal()), graph)
        else:
            state = tuple(state)
            graph = self._graph_t(encoding=self, state=state)
            self._encode(state, graph)
        return graph

    @abstractmethod
    def _encode(
        self, items: Sequence[XAtom] | Sequence[XLiteral], graph: nx.Graph | nx.DiGraph
    ): ...

    @staticmethod
    def _contained_objects(
        items: Sequence[XAtom] | Sequence[XLiteral],
    ) -> list[Object]:
        return sorted(
            (
                gather_objects(items)
                if isinstance(items[0], XAtom)
                else gather_objects(item.atom for item in items)
            ),
            key=lambda obj: obj.get_name(),
        )

    @abstractmethod
    def to_pyg_data(self, encoded_graph: GraphT) -> pyg.data.Data | pyg.data.HeteroData:
        """
        Converts the encoded state into a torch-geometric data object.
        Parameters
        ----------
        encoded_graph: GraphT, the graph to convert.

        Returns
        -------
        pyg.data.Data
        """
        ...

    def _encoded_by_this(self, graph: GraphT) -> bool:
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


class EncoderFactory:
    def __init__(
        self, encoder_class: Type[GraphEncoderBase], kwargs: Optional[dict] = None
    ):
        self.encoder_class = encoder_class
        self.kwargs = kwargs or dict()

    def __eq__(self, other):
        if not isinstance(other, EncoderFactory):
            return NotImplemented
        return self.encoder_class == other.encoder_class and self.kwargs == other.kwargs

    def __call__(self, domain: XDomain) -> GraphEncoderBase:
        return self.encoder_class(domain, **self.kwargs)
