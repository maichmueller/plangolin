import datetime
import time
from functools import cached_property
from typing import Any, Callable, List, Optional

import lightning
import torch
import torch_geometric
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.data import Batch

from plangolin.logging_setup import get_logger
from plangolin.models.pyg_module import PyGHeteroModule, PyGModule
from plangolin.rl.panorama.validation import ValidationCallback


class ValueLearningLitModule(lightning.LightningModule):
    def __init__(
        self,
        gnn: PyGModule | PyGHeteroModule,
        valuator: torch.nn.Module,
        optim: torch.optim.Optimizer,
        loss: torch.nn.Module | None = None,
        pooling: str | Callable | torch.nn.Module = "sum",
        validation_hooks: Optional[List[ValidationCallback]] = None,
    ) -> None:
        super().__init__()
        assert isinstance(optim, torch.optim.Optimizer)
        if not isinstance(gnn, PyGHeteroModule) and not isinstance(gnn, PyGModule):
            raise ValueError(f"Unknown GNN type: {self.gnn}")
        self.gnn = gnn
        self.valuator = valuator
        self.optim = optim
        self.loss = loss or torch.nn.L1Loss(reduction="mean")
        self.validation_hooks = ModuleList(validation_hooks or [])
        if isinstance(pooling, str):
            self.pooling = {
                "sum": torch_geometric.nn.global_add_pool,
                "add": torch_geometric.nn.global_add_pool,
                "mean": torch_geometric.nn.global_mean_pool,
                "max": torch_geometric.nn.global_max_pool,
            }[pooling]
        elif isinstance(pooling, torch.nn.Module) or callable(pooling):
            self.pooling = pooling
        else:
            raise ValueError(f"Unknown state pooling arg. Got {pooling}")
        self._validation_losses = []
        self._validation_start_time = None
        self._prev_validation_dataloader_idx = -1
        self._total_val_batches: int | None = None
        self._validation_batch_counter = 0
        self._training_start_time = None

    def on_fit_start(self):
        # pass the device to the DataModule
        get_logger(__name__).info(f"using device: {self.device}")
        self.trainer.datamodule.device = self.device
        super().on_fit_start()

    def on_validation_start(self) -> None:
        n_val_loaders = len(self.trainer.val_dataloaders)
        # How many batches *per* loader
        # trainer.num_val_batches will be an int if you only
        # have one val-loader, or a list/tuple of ints if multiple.
        num_batches = self.trainer.num_val_batches

        get_logger(__name__).info(
            f"Running {n_val_loaders} validation-loaders:\n"
            + "\n".join(
                f"{i:<3}: {count} batches for {self.dataloader_names[i]}"
                for i, count in enumerate(num_batches)
            )
        )
        self._total_val_batches = sum(num_batches)
        self._validation_start_time = time.time()

    @cached_property
    def dataloader_names(self):
        return {
            i: p.name
            for i, p in enumerate(self.trainer.datamodule.data.validation_problems)
        }

    def forward(self, states: Batch, info_dict: dict[str, Any] = None):
        embedding, batch = self.gnn(states, info_dict=info_dict)
        pooled_state_embeddings = self.pooling(embedding, batch, size=states.num_graphs)
        valuation = self.valuator(pooled_state_embeddings)
        return valuation

    def on_train_epoch_start(self) -> None:
        self._training_start_time = time.time()
        super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        elapsed = time.time() - self._training_start_time
        self._training_start_time = None
        td = datetime.timedelta(seconds=elapsed)
        get_logger(__name__).info("Training epoch finished in %s", str(td))
        super().on_train_epoch_end()

    def _step(
        self,
        batch_tuple: tuple[Batch, Tensor, dict[str, Any]],
        batch_idx: int = None,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        batch, targets, info = batch_tuple
        estim_values = self(batch, info_dict=info)
        loss = self._compute_loss(estim_values, targets)
        return loss

    def training_step(
        self,
        batch_tuple: tuple[Batch, Tensor, dict[str, Any]],
        batch_idx: int = None,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        return self._step(batch_tuple, batch_idx, dataloader_idx)

    def validation_step(
        self,
        batch_tuple: tuple[Batch, Tensor, dict[str, Any]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss = self._step(batch_tuple, batch_idx, dataloader_idx)
        self._validation_log(loss, batch_tuple[2], batch_idx, dataloader_idx)

    def _compute_loss(
        self,
        estimated: Tensor,
        target: Tensor,
    ) -> Tensor:
        return self.loss(estimated, target)

    def _validation_log(self, loss, info, batch_idx, dataloader_idx):
        td = TensorDict(
            {
                self.loss.__class__.__name__: loss,
            }
        )
        for hook in self.validation_hooks:
            metrics = hook(
                td,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )
            if metrics:
                for k, v in metrics.items():
                    self.log(
                        f"val/{k}",
                        v,
                        batch_size=info["batch_size"],
                        on_epoch=True,
                    )

    def on_validation_epoch_end(self) -> None:
        elapsed = time.time() - self._validation_start_time
        td = datetime.timedelta(seconds=elapsed)
        get_logger(__name__).info("Validation epoch finished in %s", str(td))
        self._prev_validation_dataloader_idx = -1
        self._validation_batch_counter = 0

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim
