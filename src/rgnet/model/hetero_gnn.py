from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from lightning import LightningModule
from torch import Tensor
from torch_geometric.typing import Adj

from rgnet.model.hetero_message_passing import FanInMP, FanOutMP


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layer: int,
        obj_name: str,
        arity_by_pred: Dict[str, int],
    ):
        """
        :param hidden_size: The size of object embeddings.
        :param num_layer: The iterations of message exchanges.
        :param obj_name: The type name of objects in the x_dict.
        :param arity_by_pred: A dictionary mapping predicate names to their arity.
        Creates one MLP for each predicate.
        Note that predicates as well as goal-predicates are meant.
        """
        super().__init__()

        self.num_layer: int = num_layer
        self.obj_name: str = obj_name
        mlp_by_pred = {
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object..
            pred: pyg.nn.MLP(
                [
                    arity * hidden_size,
                    hidden_size,
                    arity * hidden_size,
                ],
                norm="layer_norm",  # necessary for batches of size 1
            )
            for pred, arity in arity_by_pred.items()
            if arity > 0
        }

        self.obj_to_atom = FanOutMP(mlp_by_pred, src_name=obj_name)

        self.obj_update = pyg.nn.MLP([hidden_size * 2, hidden_size * 2, hidden_size])
        self.atom_to_obj = FanInMP(hidden_size=hidden_size, dst_name=obj_name)
        self.readout = pyg.nn.MLP(
            [hidden_size, 2 * hidden_size, 1], act="Mish", norm="layer_norm"
        )

    def layer(self, x_dict, edge_index_dict):
        # Groups object embeddings that are part of an atom and
        # applies predicate-specific MLP based on the edge type.
        out = self.obj_to_atom(x_dict, edge_index_dict)
        x_dict.update(out)  # update atom embeddings
        # Distribute the atom embeddings back to the corresponding objects.
        out = self.atom_to_obj(x_dict, edge_index_dict)
        # Update the object embeddings using a shared update-MLP.
        obj_emb = torch.cat([x_dict[self.obj_name], out[self.obj_name]], dim=1)
        obj_emb = self.obj_update(obj_emb)
        x_dict[self.obj_name] = obj_emb

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):
        # Filter out dummies
        x_dict = {k: v for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.numel() != 0}

        for _ in range(self.num_layer):
            self.layer(x_dict, edge_index_dict)

        obj_emb = x_dict[self.obj_name]
        batch = (
            batch_dict[self.obj_name]
            if batch_dict is not None
            else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=obj_emb.device)
        )
        # Aggregate all object embeddings into one aggregated embedding
        aggr = pyg.nn.global_add_pool(obj_emb, batch)  # shape [hidden, 1]
        # Produce final single scalar of shape [1]
        return self.readout(aggr).view(-1)


class LightningHetero(LightningModule):

    def __init__(
        self,
        hidden_size,
        num_layer: int,
        obj_name: str,
        arity_by_pred: Dict[str, int],
        lr: float = 0.001,
        weight_decay: float = 5e-4,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = HeteroGNN(hidden_size, num_layer, obj_name, arity_by_pred)

    def forward(self, x_dic, edge_index_dict, batch_dict):
        return self.model(x_dic, edge_index_dict, batch_dict)

    def training_step(self, data, batch_index) -> torch.Tensor:
        return self._common_test_step(data, "train")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _common_test_step(self, data, phase: str):
        x_out = self.forward(data.x_dict, data.edge_index_dict, data.batch_dict)
        loss = F.mse_loss(x_out, data.y.float())
        self.log(f"{phase}_loss", loss, batch_size=data.batch_size)
        return loss

    def validation_step(self, batch, batch_index):
        return self._common_test_step(batch, "val")

    def test_step(self, batch, batch_index):
        return self._common_test_step(batch, "test")

    def num_parameter(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
