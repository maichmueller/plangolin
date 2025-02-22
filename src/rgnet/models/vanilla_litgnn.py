import torch
import torch.nn.functional as F
import torch_geometric as pyg
from lightning import LightningModule
from torch import Tensor, nn
from torch_geometric.data import Batch

from rgnet.encoding import ColorGraphEncoder
from rgnet.models import PyGModule, VanillaGNN


class LitVanillaGNN(LightningModule):
    def __init__(
        self,
        gnn: PyGModule,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.gnn = gnn
        self.readout = nn.Sequential(
            nn.Linear(gnn.hiddden_size, gnn.hiddden_size),
            nn.Tanh(),
            nn.Linear(gnn.hiddden_size, gnn.size_out),
        )
        self.verbose = verbose

    def forward(self, batch: Batch) -> Tensor:
        x, edge_index, batch_info = self.gnn.unpack(batch)
        out = self.gnn(self.linear(x.view(-1, 1)), edge_index, batch)
        # sum-up the embeddings for each graph-node -> shape [embedding_size,batch_size]
        aggregated = pyg.nn.global_add_pool(out, batch)
        # reduce from embeddings_size to one -> shape [batch_size, 1]
        out = self.readout(aggregated)
        return out.view(-1)  # shape [batch_size]

    def training_step(self, batch, batch_index) -> Tensor:
        """
        :param x: The node feature matrix of floats
        :param edge_index: The adjacency list tensor
        :param batch: Batch information mapping nodes to graphs (as in DataBatch)
        """
        out = self(batch)
        loss: Tensor = self.l1_loss(out, batch.y)
        self.log("train_loss", loss, batch_size=batch.batch_size)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def _val_test_step(self, batch, phase: str):
        x_out = self.forward(batch)
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
    import xmimir as xmi

    model = VanillaGNN(hidden_size=10, num_layer=4)
    domain = xmi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = xmi.ProblemParser("test/pddl_instances/blocks/minimal.pddl").parse(domain)

    space = xmi.StateSpace.new(problem, xmi.GroundedSuccessorGenerator(problem))

    state = get_initial_state(space)
    encoder = ColorGraphEncoder(domain)
    data = encoder.to_pyg_data(encoder.encode(state))
    output = model(data.x, data.edge_index, data.batch)
    print(output)
