import itertools
import logging
import queue
import threading
import time
from contextlib import ExitStack
from functools import cached_property
from typing import Any, List, NamedTuple, Optional

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.data import Batch

from rgnet.logging_setup import tqdm
from rgnet.models import HeteroGNN
from rgnet.models.atom_valuator import AtomValuator
from rgnet.models.pyg_module import PyGHeteroModule, PyGModule
from rgnet.rl.thundeRL.validation import ValidationCallback
from rgnet.utils.inference_worker import (
    InferenceProcessWorker,
    LoadWeights,
    OutputBatch,
    ProcessBatch,
    SentinelType,
    TagCompletionType,
    TagException,
)
from rgnet.utils.reshape import unsqueeze_right_like


class OutputInfo(NamedTuple):
    state_index: int
    atom: str


def batch_fn(
    model,
    batch: Batch,
    *args,
):
    return model(batch, provide_output_metadata=False)


class EmbeddingAndValuator(torch.nn.Module):
    def __init__(self, gnn: PyGModule, atom_valuator: AtomValuator):
        super().__init__()
        self.gnn = gnn
        self.atom_valuator = atom_valuator

    def forward(
        self,
        batch: Batch,
        provide_output_metadata: bool | None = None,
        info_dict: Optional[dict[str, Any]] = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, list[OutputInfo]]]:
        embeddings, batch_info = self.gnn(batch)
        return self.atom_valuator(
            embeddings,
            batch_info,
            batch.object_names,
            provide_output_metadata=provide_output_metadata,
            info_dict=info_dict,
        )


# torch.multiprocessing.set_sharing_strategy("file_system")


class AtomValuesLitModule(lightning.LightningModule):
    def __init__(
        self,
        gnn: PyGModule | PyGHeteroModule,
        atom_valuator: AtomValuator,
        optim: torch.optim.Optimizer,
        validation_hooks: Optional[List[ValidationCallback]] = None,
        normalize_loss: bool = True,
        assert_output: bool = False,
        max_queued_batches: int = 0,
        share_predicate_modules: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(optim, torch.optim.Optimizer)
        if not isinstance(gnn, PyGHeteroModule) and not isinstance(gnn, PyGModule):
            raise ValueError(f"Unknown GNN type: {self.gnn}")
        self.gnn = gnn
        self.atom_valuator = atom_valuator
        if share_predicate_modules:
            # if we want to share the MLPs, we need to ensure that the AtomValuator's MLPs are shared
            # with the GNN's predicate modules.
            if isinstance(self.gnn, HeteroGNN):
                # the valuator's mlps are a subset of the GNN's predicate modules, dont test the other way around
                valuators = atom_valuator.valuator_by_predicate
                for predicate in valuators:
                    assert isinstance(valuators[predicate], torch.nn.Sequential)
                    # replace the first module in the Sequential with the GNN's predicate module
                    valuators[predicate][0] = gnn.objects_to_atom_mp.update_modules[
                        predicate
                    ]
            else:
                logging.warning(
                    "Sharing MLPs is only supported for HeteroGNN instances. Silently ignoring."
                )
        self.embedder_and_valuator = EmbeddingAndValuator(
            gnn=self.gnn, atom_valuator=self.atom_valuator
        )
        self.optim = optim
        self.validation_hooks = ModuleList(validation_hooks or [])
        self.normalize_loss = normalize_loss
        self.assert_output = assert_output
        self._validation_losses = []
        self._cuda_streams = None
        self._cuda_pool = None
        self._in_q: torch.multiprocessing.Queue | None = None
        self._out_q: torch.multiprocessing.Queue | None = None
        self._num_workers = 0  # Default number of workers, can be overridden
        self._completion_tags_received = {i: False for i in range(self._num_workers)}
        self._collector_thread = None
        self._collector_running = False
        self._validation_outputs = []
        self._val_workers = []
        self._worker_cycler = None
        self._exit_stack = ExitStack()
        self._validation_start_time = None
        self._collector_pbar = None
        self._prev_validation_dataloader_idx = -1
        self._total_val_batches: int | None = None
        self._validation_batch_counter = 0
        # Batches that are queued for processing in separate streams
        self._queued_batches: list[ProcessBatch] = []
        self._max_queued_batches = (
            max_queued_batches  # Maximum number of batches to queue in the workers
        )

    @property
    def cuda_streams(self):
        if self._cuda_streams is None:
            self._cuda_streams = [torch.cuda.Stream(self.device) for _ in range(10)]
            self._cuda_pool = itertools.cycle(self._cuda_streams)
        return self._cuda_streams

    def next_stream(self):
        if self.device.type != "cuda":
            return None
        assert self.cuda_streams
        return next(self._cuda_pool)

    def on_fit_start(self):
        # pass the device to the DataModule
        logging.info(f"using device: {self.device}")
        self.trainer.datamodule.device = self.device

    def on_validation_start(self) -> None:
        n_val_loaders = len(self.trainer.val_dataloaders)
        # How many batches *per* loader
        # trainer.num_val_batches will be an int if you only
        # have one val-loader, or a list/tuple of ints if multiple.
        num_batches = self.trainer.num_val_batches

        logging.info(
            f"Running {n_val_loaders} validation-loaders:\n"
            + "\n".join(
                f"{i:<3}: {count} batches for {self.dataloader_names[i]}"
                for i, count in enumerate(num_batches)
            )
        )
        self._total_val_batches = sum(num_batches)
        self._validation_start_time = time.time()
        if self._num_workers > 0:
            if not self._val_workers:
                ctx = torch.multiprocessing.get_context()
                # self._in_q = ctx.Queue()
                self._out_q = ctx.Queue()
                self._exit_stack = ExitStack()
                max_queued_batches = (
                    0
                    if self._max_queued_batches == 0
                    else (self._max_queued_batches // self._num_workers)
                )
                # spin up workers'
                self.embedder_and_valuator.share_memory()
                for worker_id in range(self._num_workers):
                    worker = InferenceProcessWorker(
                        worker_id=worker_id,
                        model=self.embedder_and_valuator,
                        batch_fn=batch_fn,
                        ctx=ctx,
                        device=self.device,
                        out_q=self._out_q,
                        max_qsize=max_queued_batches,
                    )
                    self._exit_stack.enter_context(worker)
                    self._val_workers.append(worker)
                self._worker_cycler = itertools.cycle(self._val_workers)
            else:
                # if we already have workers, we can just load the new weights
                new_state_dict = self.embedder_and_valuator.state_dict()
                for worker in self._val_workers:
                    worker.submit(LoadWeights(new_state_dict))
            # start a *thread* to collect + compute + log
            self._collector_running = True
            self._collector_thread = threading.Thread(
                target=self._validation_step_collector_loop, daemon=True
            )
            self._collector_thread.start()

    def forward(
        self,
        states_batch: Batch,
        provide_output_metadata: bool | None = None,
        info_dict: Optional[dict[str, Any]] = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, list[OutputInfo]]]:
        """
        Batch is a PyG Batch object containing the graphs of batch_size many states.

        We compute the embeddings of the objects in the states and split up the tensor into state-respective
        object-embedding pieces, i.e. each state has its own object-embedding 2D-tensor where dim-0 is the object
        dimension and dim-1 the feature dimension.
        We then compute all possible permutations of object-embeddings of size `arity(predicate)` in each state
        and batch them together again. The predicate-specific MLP then performs a readout of the atom values.
        """
        if provide_output_metadata is None:
            provide_output_metadata = self.assert_output
        return self.embedder_and_valuator(
            states_batch, provide_output_metadata, info_dict=info_dict
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
        batch_tuple: tuple[
            Batch, dict[str, Tensor], dict[str, list[tuple[int, str]]], dict[str, Any]
        ],
        batch_idx: int = None,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        batch, targets, predicate_to_state_idx, info = batch_tuple
        out_by_predicate = self(
            batch, provide_output_metadata=self.assert_output, info_dict=info
        )
        if self.assert_output:
            out_by_predicate, output_info = out_by_predicate
            self._assert_output(out_by_predicate, output_info, predicate_to_state_idx)
        assert out_by_predicate.keys() == targets.keys()
        loss, norm_loss, loss_by_predicate, norm_loss_by_predicate = self._compute_loss(
            out_by_predicate, targets
        )
        self.log(
            "train/" + f"direct_l1{' (unused)' if self.normalize_loss else ''}",
            loss.item(),
            batch_size=batch.batch_size,
        )
        self.log(
            "train/" + f"normalized_l1{' (unused)' if not self.normalize_loss else ''}",
            norm_loss.item(),
            batch_size=batch.batch_size,
        )
        if self.normalize_loss:
            return norm_loss
        else:
            return loss

    def _compute_loss(
        self,
        out_by_predicate: dict[str, Tensor],
        targets: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], dict[str, Tensor]]:
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction type: {reduction}")

        torch_reduce = getattr(torch, reduction)
        loss_by_predicate: dict[str, Tensor] = {}
        norm_loss_by_predicate: dict[str, Tensor] = {}

        total_loss = torch.tensor(0.0, device=self.device)
        total_norm_loss = torch.tensor(0.0, device=self.device)
        total_elements = 0

        for predicate, target_values in targets.items():
            out = out_by_predicate.get(predicate)
            if out is None:
                raise KeyError(f"Missing output for predicate: {predicate}")

            # Ensure shape compatibility
            target_values = unsqueeze_right_like(target_values, out)
            # Normalize by sqrt of absolute target (add epsilon to prevent division by zero)
            epsilon = 1e-8
            normalization_values = (
                torch.clamp_min(target_values.detach().abs(), 1).sqrt() + epsilon
            )
            l1loss = torch.nn.functional.l1_loss(out, target_values, reduction="none")
            norm_l1loss = l1loss / normalization_values

            # Reduce per predicate (prevents high-arity predicates - those with many permutations - from dominating)
            reduced_loss = torch_reduce(l1loss)
            reduced_norm_loss = torch_reduce(norm_l1loss)

            loss_by_predicate[predicate] = reduced_loss
            norm_loss_by_predicate[predicate] = reduced_norm_loss

            total_loss += reduced_loss
            total_norm_loss += reduced_norm_loss
            total_elements += l1loss.numel()

        num_predicates = len(loss_by_predicate)
        if reduction == "mean":
            return (
                total_loss / num_predicates,
                total_norm_loss / num_predicates,
                loss_by_predicate,
                norm_loss_by_predicate,
            )
        else:  # "sum"
            return (
                total_loss / total_elements,
                total_norm_loss / total_elements,
                loss_by_predicate,
                norm_loss_by_predicate,
            )

    @cached_property
    def dataloader_names(self):
        return {
            i: p.name
            for i, p in enumerate(self.trainer.datamodule.data.validation_problems)
        }

    def validation_step(
        self,
        batch_tuple: tuple[
            Batch,
            dict[str, torch.Tensor],
            dict[str, list[tuple[int, str]]],
            dict[str, Any],
        ],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        We only submit into the in_q here.  Everything else
        (receive -> compute loss -> log / hooks) is done
        asynchronously in _collector_loop().
        """
        self._validation_batch_counter += 1
        if self._num_workers == 0:
            self._enqueue_batch(ProcessBatch(batch_tuple, batch_idx, dataloader_idx))
        else:
            next(self._worker_cycler).submit(
                ProcessBatch(
                    batch=batch_tuple,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                )
            )

    def _enqueue_batch(
        self,
        batch: ProcessBatch,
    ) -> None:
        is_new_dataloader = batch.dataloader_idx != self._prev_validation_dataloader_idx
        if (
            len(self._queued_batches) >= len(self.cuda_streams)
            or is_new_dataloader
            or self._validation_batch_counter >= self._total_val_batches
        ):
            if is_new_dataloader:
                # if we are switching dataloaders, we need to clear the queued batches
                self._prev_validation_dataloader_idx = batch.dataloader_idx
            outputs = [
                self._local_validation_step(
                    proc_batch.batch,
                    proc_batch.batch_idx,
                    proc_batch.dataloader_idx,
                )
                for proc_batch in self._queued_batches
            ]
            if self.device.type == "cuda":
                # synchronize the CUDA streams to ensure all operations are completed
                for stream in self.cuda_streams:
                    stream.synchronize()
            for out, inp in zip(outputs, self._queued_batches):
                output, (batch_tuple, batch_idx, dataloader_idx) = out, inp
                targets, predicate_to_state_idx, info = batch_tuple[1:]
                self._validation_loss_and_log(
                    output,
                    targets,
                    info,
                    batch_idx,
                    dataloader_idx,
                )
            self._queued_batches.clear()
        self._queued_batches.append(batch)

    def _local_validation_step(self, batch_tuple, batch_idx, dataloader_idx):
        with torch.cuda.stream(self.next_stream()):
            batch, targets, predicate_to_state_idx, info = batch_tuple
            output = self(
                batch, provide_output_metadata=self.assert_output, info_dict=info
            )
            if self.assert_output:
                output, output_info = output
                self._assert_output(
                    output,
                    output_info,
                    predicate_to_state_idx,
                )
            return output

    def _validation_step_collector_loop(self) -> None:
        """
        Runs in its own thread.  Grabs model outputs off self._out_q,
        """
        while self._collector_running:
            try:
                # this call here is why we use a thread instead of a process to collect: despite Python's GIL, '
                # the get function is implemented in C and actually releases the GIL since no other python objects
                # are touched during the wait for the queue. This allows other threads to run in the meantime.
                item = self._out_q.get(timeout=0.0)
            except queue.Empty:
                continue
            match item:
                case SentinelType():
                    break
                case (int(), TagCompletionType()):
                    worker_id = item[0]
                    self._completion_tags_received[worker_id] = True
                    if all(self._completion_tags_received.values()):
                        self._collector_running = False
                        break
                case OutputBatch():
                    if self._collector_pbar is not None:
                        self._collector_pbar.update(1)
                    try:
                        (
                            tag,
                            output_tuple,
                            batch_idx,
                            dataloader_idx,
                        ) = item
                    except TypeError as e:
                        raise RuntimeError(
                            "Expected item from _out_q to be a tuple of (tag, output_tuple, batch_idx, dataloader_idx), "
                            f"but got: {item!r}"
                        ) from e
                    if tag is TagException:
                        raise output_tuple  # re-raise the exception

                    (
                        out_by_predicate,
                        targets,
                        predicate_to_state_idx,
                        info,
                    ) = output_tuple

                    self._validation_loss_and_log(
                        out_by_predicate, targets, info, batch_idx, dataloader_idx
                    )

    def _validation_loss_and_log(
        self, out_by_predicate, targets, info, batch_idx, dataloader_idx
    ):
        (
            loss,
            norm_loss,
            loss_by_predicate,
            norm_loss_by_predicate,
        ) = self._compute_loss(out_by_predicate, targets)
        td = TensorDict(
            {
                f"{p}": {"l1": loss, "norm_l1": norm_loss_by_predicate[p]}
                for p, loss in loss_by_predicate.items()
            }
            | {
                "l1": loss,
                "norm_l1": norm_loss,
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
        self._prev_validation_dataloader_idx = -1
        self._validation_batch_counter = 0
        if self._num_workers > 0:
            self._collector_pbar = tqdm(
                range(sum(worker.in_q.qsize() for worker in self._val_workers)),
                desc="Finishing validation outputs of last dataloader",
            )
            for worker in self._val_workers:
                worker.submit(TagCompletionType())
            # 1) stop the collector thread
            self._collector_thread.join()
            self._validation_losses.clear()
            self._completion_tags_received = {
                i: False for i in range(self._num_workers)
            }
            self._collector_pbar = None
        else:
            for out in self._validation_outputs:
                self._validation_loss_and_log(*out)
            self._validation_outputs.clear()

    def teardown(self, stage: str) -> None:
        # in case Lightning calls teardown after exception, ensure we clean up
        # shut down our workers and cleanup
        if self._exit_stack:
            self._exit_stack.close()
        self._val_workers.clear()
        if self._collector_thread and self._collector_thread.is_alive():
            self._collector_running = False
            self._collector_thread.join()
        super().teardown(stage)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optim
