from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch_geometric as pyg
import wandb
from lightning import LightningModule
from torch import Tensor
from torch.nn import L1Loss, MSELoss
from torch.nn.modules.loss import _Loss

from .hetero_gnn import ValueHeteroGNN


class LightningHetero(LightningModule):
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_WEIGHT_DECAY = 5e-4
    DEFAULT_LOSS = "L1Loss"
    DEFAULT_HIDDEN_SIZE = 32
    DEFAULT_NUM_LAYER = 30
    DEFAULT_AGGREGATION = "sum"

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        loss_function: _Loss | str,
        hidden_size: int,
        num_layer: int,
        aggregation: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
    ) -> None:
        super().__init__()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.loss_function: L1Loss
        if loss_function == "l1":
            self.loss_function = L1Loss()
        elif loss_function == "mse":
            self.loss_function = MSELoss()
        elif not isinstance(loss_function, _Loss):
            raise ValueError(f"Unknown loss function: {loss_function}")

        self.model = ValueHeteroGNN(
            hidden_size,
            num_layer=num_layer,
            obj_type_id=obj_type_id,
            arity_dict=arity_dict,
            aggr=aggregation,
        )
        self.save_hyperparameters()
        self.val_loss_by_label: Dict[int, List[Tensor]] = defaultdict(list)

    @staticmethod
    def add_model_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("HeteroGNN")
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=LightningHetero.DEFAULT_HIDDEN_SIZE,
            help=f"Embedding size of graph nodes (default: {LightningHetero.DEFAULT_HIDDEN_SIZE})",
        )
        parser.add_argument(
            "--num_layer",
            type=int,
            default=LightningHetero.DEFAULT_NUM_LAYER,
            help=f"Number of message-passing layer (default: {LightningHetero.DEFAULT_NUM_LAYER})",
        )
        parser.add_argument(
            "--lr",
            dest="learning_rate",
            type=float,
            default=LightningHetero.DEFAULT_LEARNING_RATE,
            help=f"Learning rate used by Adam (default: {LightningHetero.DEFAULT_LEARNING_RATE})",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=LightningHetero.DEFAULT_WEIGHT_DECAY,
            help=f"Weight decay used by Adam (default: {LightningHetero.DEFAULT_WEIGHT_DECAY})",
        )
        parser.add_argument(
            "--loss",
            dest="loss_function",
            choices=["l1", "mse"],
            type=str,
            default=LightningHetero.DEFAULT_LOSS,
            help=f"Loss function (default: {LightningHetero.DEFAULT_LOSS})",
        )
        parser.add_argument(
            "--aggregation",
            choices=["sum", "mean", "softmax"],
            type=str,
            default=LightningHetero.DEFAULT_AGGREGATION,
            help=f"Message-aggregation function (default: {LightningHetero.DEFAULT_AGGREGATION})",
        )
        return parent_parser

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

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
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
