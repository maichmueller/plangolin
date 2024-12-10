import torch
import torch.nn.functional as F
import torch_geometric as pyg
from lightning import LightningModule
from torch import Tensor, nn
from torch.nn import LayerNorm, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv

from rgnet.encoding import ColorGraphEncoder
from rgnet.models import VanillaGNN


class LitVanillaGNN(LightningModule):
    def __init__(
        self,
        vanilla_gnn: VanillaGNN,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.vanilla_gnn = vanilla_gnn
        self.readout = nn.Sequential(
            nn.Linear(vanilla_gnn.hiddden_size, vanilla_gnn.hiddden_size),
            nn.Tanh(),
            nn.Linear(vanilla_gnn.hiddden_size, vanilla_gnn.size_out),
        )
        self.verbose = verbose

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
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

    def training_step(self, batch, batch_index) -> Tensor:
        x, edge_index = batch.x, batch.edge_index
        out = self(x, edge_index, batch.batch)
        loss: Tensor = self.l1_loss(out, batch.y)
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

    def on_after_backward(self):
        if self.verbose and self.trainer.global_step % 1 == 0:
            grad_vec = None
            for p in self.parameters():
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))

            self.log(
                "Cum. abs. gradient", grad_vec.abs().sum(), on_step=True, on_epoch=False
            )


if __name__ == "__main__":
    import pymimir as mi

    model = VanillaGNN(size_out=1, size_in=1, size_embedding=10, num_layer=4)
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/minimal.pddl").parse(domain)

    space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))

    state = space.get_initial_state()
    encoder = ColorGraphEncoder(domain)
    data = encoder.to_pyg_data(encoder.encode(state))
    out = model(data.x, data.edge_index, data.batch)
    print(out)
