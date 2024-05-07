from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch_geometric as pyg
import wandb
from lightning import LightningModule
from torch import Tensor
from torch.nn.modules.loss import L1Loss, MSELoss, _Loss
from torch_geometric.typing import Adj

from rgnet.models.hetero_message_passing import FanInMP, FanOutMP


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        obj_type_id: str,
        arity_dict: Dict[str, int],
    ):
        """
        :param hidden_size: The size of object embeddings.
        :param num_layer: Total number of message exchange iterations.
        :param obj_type_id: The type identifier of objects in the x_dict.
        :param arity_dict: A dictionary mapping predicate names to their arity.
        Creates one MLP for each predicate.
        Note that predicates as well as goal-predicates are meant.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layer: int = num_layer
        self.obj_type_id: str = obj_type_id
        mlp_dict = {
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            pred: HeteroGNN.mlp(
                hidden_size * arity, hidden_size * arity, hidden_size * arity
            )
            for pred, arity in arity_dict.items()
            if arity > 0
        }

        self.obj_to_atom = FanOutMP(mlp_dict, src_name=obj_type_id)

        self.obj_update = HeteroGNN.mlp(
            in_size=2 * hidden_size, hidden_size=2 * hidden_size, out_size=hidden_size
        )

        self.atom_to_obj = FanInMP(hidden_size=hidden_size, dst_name=obj_type_id)
        self.readout = HeteroGNN.mlp(hidden_size, 2 * hidden_size, 1)

    def encoding_layer(self, x_dict: Dict[str, Tensor]):
        # Resize everything by the hidden_size
        # embedding of objects = hidden_size
        # embedding of atoms = arity of predicate * hidden_size
        for k, v in x_dict.items():
            assert v.dim() == 2
            x_dict[k] = torch.zeros(
                v.shape[0], v.shape[1] * self.hidden_size, device=v.device
            )
        return x_dict

    def layer(self, x_dict, edge_index_dict):
        # Groups object embeddings that are part of an atom and
        # applies predicate-specific MLP based on the edge type.
        out = self.obj_to_atom(x_dict, edge_index_dict)
        x_dict.update(out)  # update atom embeddings
        # Distribute the atom embeddings back to the corresponding objects.
        out = self.atom_to_obj(x_dict, edge_index_dict)
        # Update the object embeddings using a shared update-MLP.
        obj_emb = torch.cat([x_dict[self.obj_type_id], out[self.obj_type_id]], dim=1)
        obj_emb = self.obj_update(obj_emb)
        x_dict[self.obj_type_id] = obj_emb

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):
        # Filter out dummies
        x_dict = {k: v for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.numel() != 0}

        x_dict = self.encoding_layer(x_dict)  # Resize everything by the hidden_size

        for _ in range(self.num_layer):
            self.layer(x_dict, edge_index_dict)

        obj_emb = x_dict[self.obj_type_id]
        batch = (
            batch_dict[self.obj_type_id]
            if batch_dict is not None
            else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=obj_emb.device)
        )
        # Aggregate all object embeddings into one aggregated embedding
        aggr = pyg.nn.global_add_pool(obj_emb, batch)  # shape [hidden, 1]
        # Produce final single scalar of shape [1]
        return self.readout(aggr).view(-1)

    @staticmethod
    def mlp(in_size: int, hidden_size: int, out_size: int):
        return pyg.nn.MLP([in_size, hidden_size, out_size], norm=None, dropout=0.0)


class LightningHetero(LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        loss_function: _Loss | None = None,  # default l1 loss
        **kwargs,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_function = loss_function or L1Loss()
        self.model = HeteroGNN(**kwargs)
        self.save_hyperparameters()
        self.val_loss_by_label: Dict[int, List[Tensor]] = defaultdict(list)

    def loss(self, out, true_ys):
        exp_label = (
            true_ys.float()
            if (
                isinstance(self.loss_function, MSELoss)
                and (true_ys.dtype == torch.int or true_ys.dtype == torch.long)
            )
            else true_ys
        )
        return self.loss_function(out, exp_label)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        return self.model(x_dict, edge_index_dict, batch_dict)

    def training_step(self, data, batch_index) -> torch.Tensor:
        return self._common_step(data, "train")[1]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def on_validation_epoch_start(self) -> None:
        self.val_loss_by_label.clear()

    def _common_step(self, batch, phase: str):
        out = self.forward(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        loss = self.loss(out, batch.y)
        self.log(f"{phase}_loss", loss, batch_size=batch.batch_size)
        return out, loss

    def validation_step(self, batch, batch_index):
        out, val_loss = self._common_step(batch, "val")
        for i, true_y in enumerate(batch.y):
            true_y_key = true_y.item() if isinstance(true_y, torch.Tensor) else true_y
            self.val_loss_by_label[true_y_key].append(self.loss(out[i], true_y).item())
        return val_loss

    def on_validation_epoch_end(self) -> None:
        if wandb.run is None:
            return
        wandb.log(
            {
                f"val_loss/{label}": torch.tensor(losses).mean()
                for label, losses in self.val_loss_by_label.items()
            }
        )

    def test_step(self, batch, batch_index):
        return self._common_step(batch, "test")[1]

    def num_parameter(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
