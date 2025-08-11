from __future__ import annotations

import functools
import inspect
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Generic, Iterable, Optional, Sequence, Type, TypeVar, get_args

import networkx as nx
import torch_geometric as pyg

from xmimir import Object, XAtom, XDomain, XLiteral, XState, gather_objects

GraphT = TypeVar("GraphT", nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


class GraphEncoderBase(ABC, Generic[GraphT]):
    """
    Base class for encoders that map planning states to NetworkX graphs and back.

    Subclasses must specify the concrete graph type via `GraphEncoderBase[nx.GraphType]` and implement `_encode` and
    `to_pyg_data`.
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
        """Return the planning domain associated with this encoder."""
        return self._domain

    @abstractmethod
    def __eq__(self, other): ...

    def encode(self, state: XState | Iterable[XAtom | XLiteral], **kwargs) -> GraphT:
        """
        Encode a state (or explicit atoms/literals) into a NetworkX graph of type `GraphT`.

        Parameters
        ----------
        state : XState | Iterable[XAtom | XLiteral]
            Either an `XState` or an iterable of atoms/literals.

        Returns
        -------
        GraphT
            The encoded graph. For `XState`, goal literals are appended to the atoms prior to encoding.
        """

        if isinstance(state, XState):
            graph = self._graph_t(encoding=self, state=state)
            self._encode(
                tuple(chain(state.atoms(with_statics=True), state.problem.goal())),
                graph,
            )
        else:
            state = tuple(state)
            graph = self._graph_t(encoding=self, state=state)
            self._encode(state, graph)
        return graph

    @abstractmethod
    def _encode(
        self, items: Sequence[XAtom | XLiteral], graph: nx.Graph | nx.DiGraph
    ): ...

    @staticmethod
    def _contained_objects(
        items: Sequence[XAtom | XLiteral],
    ) -> list[Object]:
        """
        Collect and lexicographically sort all objects appearing in the given atoms/literals.
        """
        if len(items) == 0:
            warnings.warn(
                "Received empty atom/literal sequence. Check if the state or goal is empty."
            )
            return []
        return sorted(
            gather_objects(
                [item if isinstance(item, XAtom) else item.atom for item in items]
            ),
            key=lambda obj: obj.get_name(),
        )

    @abstractmethod
    def to_pyg_data(self, encoded_graph: GraphT) -> pyg.data.Data | pyg.data.HeteroData:
        """
        Convert an encoded graph into a PyG data object.

        Parameters
        ----------
        encoded_graph : GraphT
            The graph to convert.

        Returns
        -------
        torch_geometric.data.Data | torch_geometric.data.HeteroData
            A homogeneous or heterogeneous PyG graph, depending on the encoder.
        """
        ...

    def _encoded_by_this(self, graph: GraphT) -> bool:
        return (
            hasattr(graph, "graph")
            and "encoding" in graph.graph
            and graph.graph["encoding"] == self
        )

    def as_factory(self) -> EncoderFactory:
        """
        Return a lightweight factory that recreates this encoder without copying the domain.

        Useful for multiprocessing or (de)serialization of encoder configuration.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `as_factory` method."
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
        signature = inspect.signature(encoder_class.__init__)
        actual_kwargs = dict()
        for kwarg in kwargs or dict():
            if kwarg not in ("self", "domain") and kwarg in signature.parameters.keys():
                param = signature.parameters[kwarg]
                if (
                    param.default is not inspect.Parameter.empty
                    and kwargs[kwarg] != param.default
                ):
                    actual_kwargs[kwarg] = kwargs[kwarg]
        self.kwargs = actual_kwargs

    def __eq__(self, other):
        if not isinstance(other, EncoderFactory):
            return NotImplemented
        return self.encoder_class == other.encoder_class and self.kwargs == other.kwargs

    def __call__(self, domain: XDomain) -> GraphEncoderBase:
        return self.encoder_class(domain, **self.kwargs)

    def __str__(self):
        return f"{self.encoder_class.__name__}({self.kwargs})"
