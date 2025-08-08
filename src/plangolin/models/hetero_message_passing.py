from __future__ import annotations

import abc
import itertools
import operator
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch_geometric as pyg
import torch_geometric.nn
from torch import Tensor
from torch_geometric.nn import Aggregation, SimpleConv
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType, OptPairTensor

from plangolin.logging_setup import get_logger
from plangolin.models.logsumexp_aggr import LogSumExpAggregation
from plangolin.models.mixins import DeviceAwareMixin
from plangolin.models.patched_module_dict import PatchedModuleDict
from plangolin.utils.misc import stream_context


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
                    get_logger(__name__).warning(
                        "Failed to resolve aggregation: " + aggr
                    )
                self.aggr = aggr
        else:
            self.aggr = aggr

    @abc.abstractmethod
    def _accepts_edge(self, edge_type: EdgeType) -> bool: ...

    @abc.abstractmethod
    def _internal_forward(self, x, edges_index, edge_type: EdgeType, **kwargs): ...

    def forward(self, x_dict, edge_index_dict, **kwargs) -> Dict[str, Tensor]:
        """
        Apply message passing to each edge_index key if the edge-type is accepted.

        Calls the internal forward with a normal homogenous signature of x, edge_index

        :param x_dict: Dictionary with a feature matrix for each node type
        :param edge_index_dict: One edge_index adjacency matrix for each edge type.
        :return: Dictionary with each processed dst as key and their updated embedding as value.
        """
        out_dict: Dict[str, Any] = dict()
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
                raise KeyError(
                    f"Neither src ({src}) nor destination ({dst}) found in x_dict ({x_dict})"
                )
            out = self._internal_forward(
                x, edge_index_dict[edge_type], edge_type, **kwargs
            )
            if dst not in out_dict:
                dst_list = out_dict[dst] = []
            else:
                dst_list = out_dict[dst]
            dst_list.append(out)
        return self._group_output(out_dict, **kwargs)

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
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


class FanOutMP(DeviceAwareMixin, HeteroRouting):
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
        use_cuda_streams: bool = False,
    ) -> None:
        """ """
        super().__init__()
        self.update_modules = PatchedModuleDict(update_modules)
        self.use_cuda_streams = torch.cuda.is_available() and use_cuda_streams
        self._cuda_streams = None
        self._cuda_pool = None
        # simple conv merely groups the messages of source nodes to target nodes without modifying them.
        self.group_incoming_features = SimpleConv()
        self.src_type = src_type

    @property
    def cuda_streams(self):
        if (
            self.use_cuda_streams
            and self._cuda_streams is None
            and self.device.type == "cuda"
        ):
            self._cuda_streams = [
                torch.cuda.Stream(self.device) for _ in range(len(self.update_modules))
            ]
            self._cuda_pool = itertools.cycle(self._cuda_streams)
        return self._cuda_streams

    def next_stream(self):
        if not self.use_cuda_streams or self.device.type != "cuda":
            return None
        assert self.cuda_streams
        return next(self._cuda_pool)

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        src, *_ = edge_type
        return src == self.src_type

    def _internal_forward(self, x, edge_index, edge_type: EdgeType, **kwargs):
        position = int(edge_type[1])
        out = self.group_incoming_features(x, edge_index)
        return position, out

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
        grouped = dict()
        for predicate, value in out_dict.items():
            sorted_out = sorted(value, key=operator.itemgetter(0))
            stacked = torch.cat(tuple(out for _, out in sorted_out), dim=1)
            update_module = self.update_modules[predicate]
            with stream_context(self.next_stream()):
                grouped[predicate] = update_module(stacked)
        self._sync_streams()
        return grouped

    def _sync_streams(self):
        if self.use_cuda_streams and (cuda_streams := self.cuda_streams) is not None:
            for stream in cuda_streams:
                stream.synchronize()


class ConditionalFanOutMP(FanOutMP):
    def _group_output(
        self, out_dict: Dict[str, List], condition: Tensor = None, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Group the output by predicate and apply the update module to each predicate's messages.

        Also requires a condition tensor that is concatenated to the messages at the end.
        """
        grouped = dict()
        for predicate, value in out_dict.items():
            sorted_out = sorted(value, key=operator.itemgetter(0))
            stacked = torch.cat(
                tuple(itertools.chain((out for _, out in sorted_out), condition)), dim=1
            )
            update_module = self.update_modules[predicate]
            with stream_context(self.next_stream()):
                grouped[predicate] = update_module(stacked)
        self._sync_streams()
        return grouped


class FanInMP(HeteroRouting):
    """ """

    def __init__(
        self,
        embedding_size: int,
        dst_name: str,
        aggr: str | torch_geometric.nn.Aggregation | None = None,
    ) -> None:
        aggr = aggr or LogSumExpAggregation()
        super().__init__(aggr)
        self.select = SelectMP(embedding_size)
        self.dst_name = dst_name

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        *_, dst = edge_type
        return dst == self.dst_name

    def _internal_forward(self, x, edges_index, edge_type, **kwargs):
        return self.select(x, edges_index, int(edge_type[1]))

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
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
        embedding_size: int,
        aggr: Optional[str | List[str]] = "sum",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.embedding_size = embedding_size

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, position: int
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x, position=position)

    def message(self, x_j: Tensor, position: int = None) -> Tensor:
        # Take the i-th hidden-number of elements from the last dimension
        # e.g from [1, 2, 3, 4, 5, 6] with hidden=2 and position=1 -> [3, 4]
        # alternatively:
        #   split = torch.split(x_j, self.embedding_size, dim=-1)
        #   return split[position]
        sliced = x_j[
            ..., position * self.embedding_size : (position + 1) * self.embedding_size
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
