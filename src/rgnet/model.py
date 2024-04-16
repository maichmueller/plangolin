import torch_geometric as pyg
from torch import nn


class PureGNN(nn.Module):

    def __init__(self, in_channel: int, embedding_size: int, num_layer: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channel, embedding_size)
        self.layer = nn.ModuleList(
            [
                pyg.nn.GCNConv(embedding_size, embedding_size, node_dim=0)
                for _ in range(num_layer)
            ]
        )
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
        for gcn_layer in self.layer:
            x = gcn_layer(x, edge_index)
            x.relu()
        # sum-up the embeddings for each graph-node -> shape [embedding_size,batch_size]
        aggregated = pyg.nn.global_add_pool(x, batch)
        # reduce from embeddings_size to one -> shape [batch_size, 1]
        return self.readout(aggregated)

    def num_parameter(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
