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


class AtomValuesLitModule(lightning.LightningModule):
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

    @staticmethod
    def _assert_output(
        out_by_predicate: dict[str, Tensor],
        output_info: dict[str, list[OutputInfo]],
        predicate_to_state_idx: dict[str, list[tuple[int, str]]],
    ):
        for predicate, expected_output_info in predicate_to_state_idx.keys():
            assert expected_output_info == output_info[predicate], (
                "Batched output information does not match target information. The data is malaligned. "
                "Check the data-collate function for the batch building logic per predicate for deviations "
                "from the logic within this module."
            )
            assert out_by_predicate[predicate].shape[0] == len(expected_output_info), (
                "Batched output information batch size does not match target information batch size. "
                "The data is malaligned. "
                "Check the data-collate function for the batch building logic per predicate for deviations "
                "from the logic within this module."
            )

    def training_step(
        self,
        batch: Batch,
        targets: dict[str, Tensor],
        predicate_to_state_idx: dict[str, list[tuple[int, str]]],
    ) -> STEP_OUTPUT:
        out_by_predicate = self(batch, with_info=self.assert_output)
        if self.assert_output:
            self._assert_output(*out_by_predicate, predicate_to_state_idx)
        assert out_by_predicate.keys() == targets.keys()
        loss, norm_loss, loss_by_predicate = self._compute_loss(
            out_by_predicate, targets
        )
        self.log(
            "train/" + f"direct_l1{' (unused)' if self.normalize_loss else ''}",
            loss.item(),
            batch_size=batch[0].batch_size,
        )
        self.log(
            "train/" + f"normalized_l1{' (unused)' if not self.normalize_loss else ''}",
            norm_loss.item(),
            batch_size=batch[0].batch_size,
        )
        if self.normalize_loss:
            return norm_loss
        else:
            return loss

    def _compute_loss(
        self,
        out_by_predicate: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        loss_by_predicate: dict[str, Tensor] = dict()
        for predicate, target_values in targets.keys():
            out = out_by_predicate[predicate]
            l1loss = torch.abs(out - target_values)
            loss_by_predicate[predicate] = l1loss
        # Normalize the loss by the square root of the target values.
        # This diminishes the weight of atoms with large values (e.g., when they are distances).
        loss = torch.zeros(1, device=self.device)
        norm_loss = torch.zeros(1, device=self.device)
        nr_predicates = len(loss_by_predicate)
        for predicate, target_values in targets.items():
            pred_loss = loss_by_predicate[predicate]
            # Take the mean for each predicate individually, since `sum` we would skew the loss towards higher arity
            # predicates. Higher arity predicates will have more atoms (permutations) to compute the loss for and
            # thus more loss values to sum up.
            # The mean per predicate will ensure that the atom count does not emphasize one predicate over another.
            avg_pred_loss = loss_by_predicate[predicate].mean()
            loss_by_predicate[predicate] = avg_pred_loss
            loss += avg_pred_loss
            # Normalize the loss by the square root of the target values.
            norm_loss += (pred_loss / target_values.detach().abs().sqrt()).mean()
        return loss / nr_predicates, norm_loss / nr_predicates, loss_by_predicate

    def validation_step(
        self,
        batch: Batch,
        targets: dict[str, Tensor],
        *args,
        batch_idx=None,
        dataloader_idx=0,
        **kwargs,
    ):
        estimated_values = self(batch, with_info=False)
        loss, norm_loss, loss_per_predicate = self._compute_loss(
            estimated_values, targets
        )
        for hook in self.validation_hooks:
            optional_metrics = hook(
                norm_loss if self.normalize_loss else loss,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )
            optional_metrics_2 = hook(
                loss_per_predicate, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
            metrics = dict()
            if optional_metrics is None:
                if optional_metrics_2 is not None:
                    metrics = optional_metrics_2
            else:
                if optional_metrics_2 is not None:
                    metrics = {**optional_metrics, **optional_metrics_2}
                else:
                    metrics = optional_metrics
            for key, value in metrics.items():
                self.log("validation/" + key, value, batch_size=batch[0].batch_size)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim
