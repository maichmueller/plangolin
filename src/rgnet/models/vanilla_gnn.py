from typing import Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import Tensor, nn
from torch.nn import LayerNorm, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.typing import Adj

from rgnet.encoding import ColorGraphEncoder
from rgnet.models.pyg_module import PyGModule
from rgnet.utils.activation_map import get_activation


class VanillaGNN(PyGModule):
    def __init__(
        self,
        size_out: int,
        size_in: int,
        hidden_size: int,
        num_layer: int,
        aggr: Optional[Union[str, pyg.nn.aggr.Aggregation]],
        pool: Optional[Union[str, Callable[[Tensor, Tensor], Tensor]]] = None,
        activation: Union[str, Callable, None] = None,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        # attrs
        self.size_out = size_out
        self.size_in = size_in
        self.hidden_size = hidden_size
        if pool is not None and isinstance(pool, str) and pool:
            if pool == "add":
                pool = pyg.nn.global_add_pool
            elif pool == "mean":
                pool = pyg.nn.global_mean_pool
            elif pool == "max":
                pool = pyg.nn.global_max_pool
            else:
                raise ValueError(
                    f"Unknown pooling function: {pool}. Choose from [add, mean, max]."
                )
        self.pool = pool
        # modules
        self.linear = nn.Linear(size_in, hidden_size)
        self.layer = nn.ModuleList()
        if activation is not None:
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
            self.layer.append(
                DeepGCNLayer(
                    conv,
                    norm=LayerNorm(hidden_size, elementwise_affine=True),
                    act=activation,
                    block="res+",
                    dropout=0.1,
                )
            )
        self.verbose = verbose

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
    ):
        """
        :param x: The node feature matrix of floats
        :param edge_index: The adjacency list tensor
        :param batch: Batch information mapping nodes to graphs (as in DataBatch)
        :return: Value estimate for every input graph, tensor of shape [32, 1]
        """
        x = self.linear(x.view(-1, 1))
        x = self.layer[0].conv(x, edge_index)
        for layer in self.layer[1:]:
            x = layer(x, edge_index)

        x = self.layer[0].act(self.layer[0].norm(x))
        # sum-up the embeddings for each graph-node -> shape [embedding_size,batch_size]
        aggregated = pyg.nn.global_add_pool(x, batch)
        # reduce from embeddings_size to one -> shape [batch_size, 1]
        out = self.readout(aggregated)
        return out.view(-1)  # shape [batch_size]


if __name__ == "__main__":
    import pymimir as mi

    model = VanillaGNN(size_out=1, size_in=1, hidden_size=10, num_layer=4)
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/minimal.pddl").parse(domain)

    space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))

    state = space.get_initial_state()
    encoder = ColorGraphEncoder(domain)
    data = encoder.to_pyg_data(encoder.encode(state))
    out = model(data.x, data.edge_index, data.batch)
    print(out)
