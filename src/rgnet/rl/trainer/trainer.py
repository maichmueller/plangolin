import pathlib
import warnings
from typing import Callable, List, Optional, Union

import torch
import torchrl.trainers
from tensordict import TensorDictBase
from torch import optim
from torchrl.objectives import LossModule
from torchrl.record.loggers import Logger
from tqdm import tqdm

from rgnet.rl import RolloutCollector


class Trainer(torchrl.trainers.Trainer):
    collector: RolloutCollector

    def __init__(
        self,
        *,
        collector: RolloutCollector,
        epochs: int,
        optim_steps_per_batch: int,
        loss_module: Union[LossModule, Callable[[TensorDictBase], TensorDictBase]],
        optimizer: Optional[optim.Optimizer] = None,
        logger: Optional[Logger] = None,
        clip_grad_norm: bool = True,
        clip_norm: float = None,
        progress_bar: bool = True,
        seed: int = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: Optional[Union[str, pathlib.Path]] = None,
        eval_hooks: List | None = None,
        eval_interval: int = 100,
    ) -> None:
        super().__init__(
            collector=collector,
            # torchrl wants to measure everything in frames, that is the total number
            # of states in which the agent made a decision.
            total_frames=collector.num_batches
            * collector.env.batch_size[0]
            * collector.rollout_length
            * epochs,
            frame_skip=1,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
        )
        self.epochs = epochs
        self._early_stopping_ops = []
        self.eval_hooks = eval_hooks
        self.eval_interval: int = eval_interval

    def register_op(self, dest: str, op: Callable, **kwargs) -> None:
        if dest == "early_stopping":
            self._early_stopping_ops.append((op, kwargs))
        else:
            super().register_op(dest, op, **kwargs)

    def _early_stopping_hook(
        self, batch: TensorDictBase, average_loss: TensorDictBase
    ) -> bool:
        should_stop = False
        for op, kwargs in self._early_stopping_ops:
            should_stop = should_stop or op(
                batch=batch, average_loss=average_loss, **kwargs
            )
        return should_stop

    def optim_steps(self, batch: TensorDictBase) -> TensorDictBase:
        """Same as parent with the addition that we return the average losses."""
        average_losses: Optional[TensorDictBase] = None

        self._pre_optim_hook()

        for j in range(self.optim_steps_per_batch):
            self._optim_count += 1

            sub_batch = self._process_optim_batch_hook(batch)
            losses_td = self.loss_module(sub_batch)
            self._post_loss_hook(sub_batch)

            losses_detached = self._optimizer_hook(losses_td)
            self._post_optim_hook()
            self._post_optim_log(sub_batch)

            if average_losses is None:
                average_losses: TensorDictBase = losses_detached
            else:
                for key, item in losses_detached.items():
                    val = average_losses.get(key)
                    average_losses.set(key, val * j / (j + 1) + item / (j + 1))
            del sub_batch, losses_td, losses_detached

        if self.optim_steps_per_batch > 0:
            self._log(
                optim_steps=self._optim_count,
                **average_losses,
            )
        return average_losses

    def train(self):
        """Same as parent with the addition of early stopping, validation and epochs."""
        if self.progress_bar:
            self._pbar = tqdm(total=self.total_frames)
            self._pbar_str = {}

        for epoch in range(self.epochs):

            self.collector.reset()

            for batch in self.collector:
                batch = self._process_batch_hook(batch)
                current_frames = (
                    batch.get(("collector", "mask"), torch.tensor(batch.numel()))
                    .sum()
                    .item()
                    * self.frame_skip
                )
                self.collected_frames += current_frames
                self._pre_steps_log_hook(batch)

                if self.collected_frames > getattr(
                    self.collector, "init_random_frames", 0
                ):
                    average_losses: TensorDictBase = self.optim_steps(batch)
                    if self._early_stopping_hook(
                        batch=batch, average_loss=average_losses
                    ):
                        self.save_trainer(force_save=True)
                        break

                self._post_steps_hook()

                self._post_steps_log_hook(batch)

                if self.progress_bar:
                    self._pbar.update(current_frames)
                    self._pbar_description()

                if self.collected_frames % self.eval_interval == 0:
                    self.validate()

            # Save trainer after each epoch
            self.save_trainer()

        self.collector.shutdown()

    def validate(self):
        if self.eval_hooks is None:
            warnings.warn(
                "Called validate without eval_collector being preset."
                "Skipping validation."
            )
            return

        for eval_hook in self.eval_hooks:
            results = eval_hook()
            if self.logger is not None:
                for key, item in results.items():
                    self.logger.log_scalar(
                        "val/" + key, item, step=self.collected_frames
                    )
