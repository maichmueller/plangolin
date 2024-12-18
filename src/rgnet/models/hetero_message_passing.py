import abc
import logging
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

    def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        aggregated: Dict[str, Tensor] = {}
        for key, value in out_dict.items():
            # hetero_conv.group does not yet support Aggregation modules
            if isinstance(self.aggr, Aggregation):
                out = torch.stack(value, dim=0)
                out = self.aggr(out, dim=0)
                if out.dim() == 3 and out.shape[0] == 1:
                    out = out[0]  # TODO Why does Softmax return one dim to much
            else:
                out = group(value, self.aggr)
            aggregated[key] = out
        return aggregated

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

            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

        return self._group_out(out_dict)


class FanOutMP(HeteroRouting):
    """
     Accepts EdgeTypes with the defined src_name.
    1. For each destination concatenate all embeddings of the source.
    2. Run the destination specific MLP on the concatenated embeddings.
    3. Save the new embedding under the destination key.
    FanOut should be aggregation free in theory.
    Every atom gets only as many messages as the arity of its predicate.
    :param update_mlp_by_dst: An MLP for each possible destination.
        Needs the degree of incoming edges as input and output dimension
    :param src_name: The node-type for which outgoing edges should be accepted.
    """

    def __init__(
        self,
        update_mlp_by_dst: Dict[str, torch.nn.Module],
        src_name: str,
    ) -> None:
        """ """
        super().__init__()
        self.update_mlp_by_dst = ModuleDict(update_mlp_by_dst)
        self.simple = SimpleConv()
        self.src_name = src_name

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        src, *_ = edge_type
        return src == self.src_type

    def _internal_forward(self, x, edge_index, edge_type: EdgeType):
        position = int(edge_type[1])
        out = self.simple(x, edge_index)
        return position, out

    def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        for dst, value in out_dict.items():
            sorted_out = sorted(value, key=lambda tpl: tpl[0])
            stacked = torch.cat([out for _, out in sorted_out], dim=1)
            out_dict[dst] = self.update_mlp_by_dst[dst](stacked)

        return out_dict


class FanInMP(HeteroRouting):

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

    def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
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
        self.hidden = hidden_size

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, position: int
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x, position=position)

    def message(self, x_j: Tensor, position: int) -> Tensor:
        # Take the i-th hidden-number of elements from the last dimension
        # e.g from [1, 2, 3, 4, 5, 6] with hidden=2 and position=1 -> [3, 4]
        # alternatively split = torch.split(x_j, self.hidden, dim=-1)
        #               split[self.position]
        sliced = x_j[:, position * self.hidden : (position + 1) * self.hidden]
        return sliced

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        return inputs, index, dim_size
