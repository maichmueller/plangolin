from __future__ import annotations

import abc
import logging
import operator
from collections import defaultdict
from functools import singledispatchmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch_geometric as pyg
import torch_geometric.nn
from torch import Tensor
from torch_geometric.nn import Aggregation, SimpleConv
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType, OptPairTensor

from rgnet.models.logsumexp_aggregation import LogSumExpAggregation


class HeteroRouting(torch.nn.Module):
    """
    Handles heterogeneous message passing very similar to pyg.nn.HeteroConv.
    Instead of specifying a convolution for each EdgeType more generic rules can be used.
    """

    def __init__(self, aggr: Optional[str | Aggregation] = None) -> None:
        super().__init__()
        if isinstance(aggr, str):
            try:
                self.aggr = aggregation_resolver(query=aggr)
            except ValueError:
                if aggr != "cat" and aggr != "stack":
                    logging.warning("Failed to resolve aggregation: " + aggr)
                self.aggr = aggr
        else:
            self.aggr = aggr

    @abc.abstractmethod
    def _accepts_edge(self, edge_type: EdgeType) -> bool: ...

    @abc.abstractmethod
    def _internal_forward(self, x, edges_index, edge_type: EdgeType): ...

    def forward(self, x_dict, edge_index_dict):
        """
        Apply message passing to each edge_index key if the edge-type is accepted.

        Calls the internal forward with a normal homogenous signature of x, edge_index

        :param x_dict: Dictionary with a feature matrix for each node type
        :param edge_index_dict: One edge_index adjacency matrix for each edge type.
        :return: Dictionary with each processed dst as key and their updated embedding as value.
        """
        out_dict: Dict[str, Any] = defaultdict(list)
        for edge_type in filter(self._accepts_edge, edge_index_dict.keys()):
            src, rel, dst = edge_type
            if src == dst and src in x_dict:
                x = x_dict[src]
            elif src in x_dict or dst in x_dict:
                x = (
                    x_dict.get(src, None),
                    x_dict.get(dst, None),
                )
            else:
                raise ValueError(
                    f"Neither src ({src}) nor destination ({dst})"
                    + f" found in x_dict ({x_dict})"
                )
            out = self._internal_forward(x, edge_index_dict[edge_type], edge_type)
            out_dict[dst].append(out)

        return self._group_output(dict(out_dict))

    def _group_output(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        aggregated: Dict[str, Tensor] = {}
        for key, value in out_dict.items():
            # `hetero_conv.group` does not yet support Aggregation modules
            if isinstance(self.aggr, Aggregation):
                out = torch.stack(value, dim=0)
                out = self.aggr(out, dim=0).squeeze(0)
            else:
                out = group(value, self.aggr)
            aggregated[key] = out
        return aggregated


class FanOutMP(HeteroRouting):
    """
    Perform the 'fanout' phase of message passing in a heterogeneous STRIPS-based graph (batch).

    Fanout refers to the number of outgoing edges of a node in the context of message passing.
    While this module can be used with generic relationships, we describe it in the STRIPS-graph case,
    the fanout refers to the first step of relational-message passing:
    Object-nodes pass their embeddings to the connected atom-nodes.

    We refer to this step also as the message-creation step of a Relational Graph Neural Network,
    since atom-embeddings store the created-messages with which object-nodes will be updated.

    Accepts `EdgeType`s whose attr `src` matches the parameter `src_type`.
    Processes the incoming edges by:
        1. For each destination, i.e. predicate, concatenate all incoming (object-)embeddings.
        2. Apply the destination specific Module to the concatenated embeddings.
        3. Save the new embedding under the destination key.

    FanOut should be aggregation free in theory.
    Every atom receives only as many messages as the arity of its predicate.

    :param update_modules: Dict, maps destination node-types to a Module (e.g. MLP) to compute messages with.
        Each Module input-and output-tensor needs to match the degree of incoming edges in shape at dim 0.
    :param src_type: The node-type whose outgoing edges should be accepted.
    """

    def __init__(
        self,
        update_modules: Dict[str, torch.nn.Module],
        src_type: str,
    ) -> None:
        """ """
        super().__init__()
        self.update_modules = ModuleDict(update_modules)
        self.static_simple_conv = SimpleConv()
        self.src_type = src_type

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        src, *_ = edge_type
        return src == self.src_type

    def _internal_forward(self, x, edge_index, edge_type: EdgeType):
        position = int(edge_type[1])
        out = self.static_simple_conv(x, edge_index)
        return position, out

    def _group_output(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        grouped = dict()
        for predicate, value in out_dict.items():
            sorted_out = sorted(value, key=operator.itemgetter(0))
            stacked = torch.cat(tuple(out for _, out in sorted_out), dim=1)
            grouped[predicate] = self.update_modules[predicate](stacked)
        return grouped


class FanInMP(HeteroRouting):
    """ """

    def __init__(
        self,
        hidden_size: int,
        dst_name: str,
        aggr: str | torch_geometric.nn.Aggregation | None = None,
    ) -> None:
        aggr = aggr or LogSumExpAggregation()
        super().__init__(aggr)
        self.select = SelectMP(hidden_size)
        self.dst_name = dst_name

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        *_, dst = edge_type
        return dst == self.dst_name

    def _internal_forward(self, x, edges_index, edge_type):
        return self.select(x, edges_index, int(edge_type[1]))

    def _group_output(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        aggregated = {}
        for dst, values in out_dict.items():
            if dst == self.dst_name:
                inputs, indices, dim_sizes = zip(*values)
                flat_inputs = torch.cat(inputs)
                flat_indices = torch.cat(indices)
                out = self.aggr(
                    x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_sizes[0]
                )
                aggregated[dst] = out
        return aggregated


class SelectMP(pyg.nn.MessagePassing):

    def __init__(
        self,
        hidden_size: int,
        aggr: Optional[str | List[str]] = "sum",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.hidden_size = hidden_size

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, position: int
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x, position=position)

    def message(self, x_j: Tensor, position: int) -> Tensor:
        # Take the i-th hidden-number of elements from the last dimension
        # e.g from [1, 2, 3, 4, 5, 6] with hidden=2 and position=1 -> [3, 4]
        # alternatively:
        #   split = torch.split(x_j, self.hidden_size, dim=-1)
        #   return split[position]
        sliced = x_j[
            ..., position * self.hidden_size : (position + 1) * self.hidden_size
        ]
        return sliced

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        return inputs, index, dim_size
