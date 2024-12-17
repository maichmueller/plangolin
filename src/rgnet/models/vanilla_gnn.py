from typing import Callable, Dict, Optional, Union

import torch_geometric as pyg
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.typing import Adj

from rgnet.encoding import ColorGraphEncoder
from rgnet.models.pyg_module import PyGModule
from rgnet.utils.activation_map import get_activation


class VanillaGNN(PyGModule):
    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        aggr: Union[str, pyg.nn.aggr.Aggregation] = "softmax",
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # attrs
        self.hidden_size = hidden_size
        # modules
        self.layers = nn.ModuleList()
        if isinstance(activation, str):
            activation = get_activation(activation, inplace=True)

        for i in range(num_layer):
            conv = GENConv(
                hidden_size,
                hidden_size,
                aggr=aggr,
                learn_t=False,
                num_layers=2,
                norm="layer",
                node_dim=0,
            )
            self.layers.append(
                DeepGCNLayer(
                    conv,
                    norm=LayerNorm(hidden_size, elementwise_affine=True),
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
    ):
        for layer in self.layers:
            x = layer(x, edge_index, batch)
        return x


if __name__ == "__main__":
    from pathlib import Path

    import pymimir as mi

    model = VanillaGNN(hidden_size=10, num_layer=4)
    data_folder = Path("data") / "pddl_domains" / "example" / "blocks"
    domain = mi.DomainParser(str(data_folder / "domain.pddl")).parse()
    problem = mi.ProblemParser(
        str(data_folder / "train" / "probBLOCKS-4-0.pddl")
    ).parse(domain)

    space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))

    state = space.get_initial_state()
    encoder = ColorGraphEncoder(domain)
    data = encoder.to_pyg_data(encoder.encode(state))
    out = model = data
    print(out)
