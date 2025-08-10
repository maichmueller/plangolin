from typing import Callable, Optional, Union

import torch_geometric as pyg
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.typing import Adj

from plangolin.models.pyg_module import PyGModule
from plangolin.utils.activation_map import get_activation


class VanillaGNN(PyGModule):
    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Union[str, pyg.nn.aggr.Aggregation] = "softmax",
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # attrs
        self.embedding_size = embedding_size
        # modules
        self.layers = nn.ModuleList()
        if isinstance(activation, str):
            activation = get_activation(activation, inplace=True)

        for i in range(num_layer):
            conv = GENConv(
                embedding_size,
                embedding_size,
                aggr=aggr,
                learn_t=False,
                num_layers=2,
                norm="layer",
                node_dim=0,
            )
            self.layers.append(
                DeepGCNLayer(
                    conv,
                    norm=LayerNorm(embedding_size, elementwise_affine=True),
                    act=activation,
                    block="res+",
                    dropout=dropout,
                )
            )

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
        **kwargs,
    ):
        for layer in self.layers:
            x = layer(x, edge_index, batch)
        return x
