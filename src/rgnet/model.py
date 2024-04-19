import lightning as L
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import Tensor, nn
from torch.nn import LayerNorm, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv


class PureGNN(L.LightningModule):

    def __init__(self, in_channel: int, embedding_size: int, num_layer: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channel, embedding_size)
        self.layer = nn.ModuleList()
        for i in range(num_layer):
            conv = GENConv(
                embedding_size,
                embedding_size,
                aggr="softmax",
                learn_t=False,
                num_layers=2,
                norm="layer",
                node_dim=0,
            )
            norm = LayerNorm(embedding_size, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block="res+", dropout=0.1)
            self.layer.append(layer)

        self.readout = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, x, edge_index, batch):
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

    def training_step(self, batch, batch_index) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        out = self(x, edge_index, batch.batch)
        loss: Tensor = F.l1_loss(out, batch.y)
        self.log("train_loss", loss, batch_size=batch.batch_size)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def _val_test_step(self, batch, phase: str):
        x, edge_index = batch.x, batch.edge_index
        x_out = self.forward(x, edge_index, batch.batch)

        loss = F.l1_loss(x_out, batch.y)
        self.log(f"{phase}_loss", loss, batch_size=batch.batch_size)
        return x_out, loss, batch.y

    def validation_step(self, batch, batch_index):
        return self._val_test_step(batch, "val")

    def test_step(self, batch, batch_index):
        return self._val_test_step(batch, "test")

    def num_parameter(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
