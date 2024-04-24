import abc
from typing import Any, Dict, List, Optional, Union

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import SimpleConv
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import Adj, EdgeType, OptPairTensor


class HeteroRouting(torch.nn.Module):
    """
    Handles heterogeneous message passing very similar to pyg.nn.HeteroConv.
    Instead of specifying a convolution for each EdgeType more generic rules can be used.
    """

    def __init__(self, aggr: Optional[str] = None) -> None:
        super().__init__()
        self.aggr = aggr

    @abc.abstractmethod
    def _accept_edge(self, src: str, rel: str, dst: str) -> bool:
        pass

    @abc.abstractmethod
    def _internal_forward(self, x, edges_index, edge_type: EdgeType):
        pass

    def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        aggregated: Dict[str, Tensor] = {}
        for key, value in out_dict.items():
            aggregated[key] = group(value, self.aggr)
        return aggregated

    def forward(self, x_dict, edge_index_dict):
        """
        Apply message passing to each edge_index key if the edge-type is accepted.
        Calls the internal forward with a normal homogenous signature of x, edge_index
        :param x_dict: Dictionary with a feature matrix for each node type
        :param edge_index_dict: One edge_index adjacency matrix for each edge type.
        :return: Dictionary with each processed dst as key and their updated embedding as value.
        """
        out_dict: Dict[str, Any] = {}
        for edge_type in edge_index_dict.keys():
            src, rel, dst = edge_type

            if not self._accept_edge(src, rel, dst):
                continue

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

    def _accept_edge(self, src, rel, dst) -> bool:
        return src == self.src_name

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
        aggr: Optional[str] = "sum",
    ) -> None:
        super().__init__(aggr)
        self.select = SelectMP(hidden_size)
        self.dst_name = dst_name

    def _accept_edge(self, src: str, rel: str, dst: str) -> bool:
        return dst == self.dst_name

    def _internal_forward(self, x, edges_index, edge_type):
        return self.select(x, edges_index, int(edge_type[1]))


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
