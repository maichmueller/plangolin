from typing import List, NamedTuple, Optional, Union

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.data import Batch

from rgnet.models.atom_valuator import AtomValuator
from rgnet.models.pyg_module import PyGHeteroModule, PyGModule
from rgnet.rl.thundeRL.validation import ValidationCallback


class OutputInfo(NamedTuple):
    state_index: int
    atom: str


class AtomValueLearningModule(lightning.LightningModule):
    def __init__(
        self,
        gnn: Union[PyGModule, PyGHeteroModule],
        atom_valuator: AtomValuator,
        optim: torch.optim.Optimizer,
        validation_hooks: Optional[List[ValidationCallback]] = None,
        normalize_loss: bool = True,
        assert_output: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(optim, torch.optim.Optimizer)
        if not isinstance(gnn, PyGHeteroModule) and not isinstance(gnn, PyGModule):
            raise ValueError(f"Unknown GNN type: {gnn}")
        self.gnn = gnn
        self.atom_valuator = atom_valuator
        self.optim = optim
        self.validation_hooks = ModuleList(validation_hooks or [])
        self.normalize_loss = normalize_loss
        self.assert_output = assert_output

    def on_fit_start(self):
        # pass the device to the DataModule
        self.trainer.datamodule.device = self.device

    def forward(
        self,
        states_batch: Batch,
        with_info: bool | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, list[OutputInfo]]]:
        """
        Batch is a PyG Batch object containing the graphs of batch_size many states.

        We compute the embeddings of the objects in the states and split up the tensor into state-respective
        object-embedding pieces, i.e. each state has its own object-embedding 2D-tensor where dim-0 is the object
        dimension and dim-1 the feature dimension.
        We then compute all possible permutations of object-embeddings of size `arity(predicate)` in each state
        and batch them together again. The predicate-specific MLP then performs a readout of the atom values.
        """
        if with_info is None:
            with_info = self.assert_output
        # embeddings shape: (batch_size * avg_object_count_per_state, embedding_size)
        embeddings, batch_info = self.gnn(states_batch)
        return self.atom_valuator(
            embeddings, batch_info, states_batch.object_names, with_info=with_info
        )

    def training_step(
        self,
        batch: Batch,
        targets: dict[str, Tensor],
        predicate_to_state_idx: dict[str, list[tuple[int, str]]],
    ) -> STEP_OUTPUT:
        out_by_predicate = self(batch)
        if self.assert_output:
            out_by_predicate, output_info = out_by_predicate
            for predicate in targets.keys():
                assert predicate_to_state_idx[predicate] == output_info[predicate], (
                    "Batched output information does not match target information. The data is malaligned. "
                    "Check the data-collate function for the batch building logic per predicate for deviations "
                    "from the logic within this module."
                )
        assert out_by_predicate.keys() == targets.keys()
        loss, loss_by_predicate = self._compute_loss(out_by_predicate, targets)
        self.log(
            "train/" + "direct_l1 (unused)",
            sum(
                map(torch.detach, loss_by_predicate.values()),
                torch.tensor(0, device=self.device),
            )
            .sum()
            .item(),
            batch_size=batch[0].batch_size,
        )
        self.log(
            "train/" + "normalized_l1", loss.item(), batch_size=batch[0].batch_size
        )
        return loss

    def _compute_loss(
        self,
        out_by_predicate: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_by_predicate: dict[str, Tensor] = dict()
        for predicate, target_values in targets.keys():
            out = out_by_predicate[predicate]
            l1loss = torch.abs(out - target_values)
            loss_by_predicate[predicate] = l1loss
        # Normalize the loss by the square root of the target values.
        # This diminishes the weight of atoms with large values (e.g. when they are distances).
        loss = torch.zeros(1, device=self.device)
        if self.normalize_loss:
            for predicate, target_values in targets.items():
                loss += loss_by_predicate[predicate] / target_values.abs().sqrt()
        else:
            for pred_loss in loss_by_predicate.values():
                loss += pred_loss
        return loss, loss_by_predicate

    def validation_step(
        self,
        batch: Batch,
        targets: dict[str, Tensor],
        predicate_to_state_idx: dict[str, list[tuple[int, str]]],
        batch_idx=None,
        dataloader_idx=0,
    ):
        estimated_values = self(batch, with_info=False)
        loss, _ = self._compute_loss(estimated_values, targets, predicate_to_state_idx)
        for hook in self.validation_hooks:
            optional_metrics = hook(
                loss,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )
            if optional_metrics is None:
                continue
            for key, value in optional_metrics.items():
                self.log("validation/" + key, value, batch_size=batch[0].batch_size)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim
