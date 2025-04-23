from __future__ import annotations

import datetime
import logging
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch_geometric.loader import ImbalancedSampler

from rgnet.encoding import GraphEncoderBase
from rgnet.rl.data_layout import InputData
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.reward import RewardFunction
from rgnet.rl.thundeRL.collate import collate_fn
from rgnet.rl.thundeRL.env_lookup import LazyEnvLookup
from rgnet.rl.thundeRL.flash_drive import FlashDrive
from rgnet.utils.utils import env_aware_cpu_count
from xmimir import Domain
from xmimir.iw import IWStateSpace


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy("file_system")


class ThundeRLDataModule(LightningDataModule):
    def __init__(
        self,
        input_data: InputData,
        reward_function: RewardFunction,
        batch_size: int,
        encoder_factory: Callable[[Domain], GraphEncoderBase],
        *,
        batch_size_validation: Optional[int] = None,
        num_workers_train: int = 6,
        num_workers_validation: int = 2,
        parallel: bool = True,
        balance_by_distance_to_goal: bool = True,
        max_cpu_count: Optional[int] = None,
        exit_after_processing: bool = False,
        flashdrive_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self._data_prepared = False
        self.data = input_data
        self.reward_function = reward_function
        self.batch_size = batch_size
        self.batch_size_validation = batch_size_validation or batch_size
        self.parallel = parallel
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_validation
        self.encoder_factory = encoder_factory
        self.balance_by_distance_to_goal = balance_by_distance_to_goal
        self.max_cpu_count = max_cpu_count
        self.exit_after_processing = exit_after_processing
        self.dataset: ConcatDataset | None = None  # late init in prepare_data()
        self.validation_sets: Sequence[Dataset] = []
        self.flashdrive_kwargs = flashdrive_kwargs or dict()
        self.envs: Mapping[Path, ExpandedStateSpaceEnv] = LazyEnvLookup(
            input_data.problem_paths + input_data.validation_problem_paths, self.get_env
        )
        # defaulted, to be overridden by trainer on setup
        self.device = torch.device("cpu")

    def get_env(self, problem: Path | int) -> ExpandedStateSpaceEnv:
        try:
            prob = self.data.problems[
                (
                    self.data.problem_paths.index(problem)
                    if isinstance(problem, Path)
                    else problem
                )
            ]
        except ValueError:
            prob = self.data.validation_problems[
                (
                    self.data.validation_problem_paths.index(problem)
                    if isinstance(problem, Path)
                    else problem
                )
            ]
        space = self.data.get_or_load_space(prob)
        if (iw_search := self.flashdrive_kwargs.get("iw_search")) is not None:
            return ExpandedStateSpaceEnv(
                IWStateSpace(
                    iw_search,
                    space,
                    **self.flashdrive_kwargs.get("iw_options", dict()),
                ),
                reward_function=self.reward_function,
                reset=True,
            )
        else:
            return ExpandedStateSpaceEnv(space, reset=True)

    def load_datasets(self, problem_paths: Sequence[Path]) -> Dict[Path, Dataset]:
        nr_total = len(problem_paths)
        completed = 0

        def update(dataset):
            nonlocal completed
            completed += 1
            logging.info(
                f"Finished loading {completed}/{nr_total} problems - Most recent loaded: {dataset.problem_path.stem} "
                f"(#{len(dataset)} states)."
            )

        datasets: Dict[Path, FlashDrive] = dict()
        flashdrive_kwargs = self.flashdrive_kwargs | dict(
            domain_path=self.data.domain_path,
            reward_function=self.reward_function,
            logging_kwargs=None,
            encoder_factory=self.encoder_factory,
        )
        start_time = time.time()
        if self.parallel and len(problem_paths) > 1:

            def enqueue_parallel(problem_path: Path, task_id: int):
                return pool.apply_async(
                    FlashDrive,
                    kwds=flashdrive_kwargs
                    | dict(
                        problem_path=problem_path,
                        root_dir=str(self.data.dataset_dir / problem_path.stem),
                        show_progress=False,
                        logging_kwargs=dict(
                            log_level=logging.root.level, task_id=task_id
                        ),
                    ),
                    callback=update,
                )

            pool_size = min(
                env_aware_cpu_count(),
                len(problem_paths),
                self.max_cpu_count or float("inf"),
            )
            with Pool(pool_size, initializer=set_sharing_strategy) as pool:
                logging.info(
                    f"Loading #{len(problem_paths)} problems in parallel using {pool_size} threads."
                )
                results = {
                    problem_path: enqueue_parallel(problem_path, i)
                    for i, problem_path in enumerate(problem_paths)
                }
                for problem_path, result in results.items():
                    datasets[problem_path] = result.get()
        else:
            for problem_path in problem_paths:
                drive = FlashDrive(
                    problem_path=problem_path,
                    root_dir=str(self.data.dataset_dir / problem_path.stem),
                    show_progress=True,
                    env=self.envs(problem_path),
                    **flashdrive_kwargs,
                )
                update(drive)
                datasets[problem_path] = drive

        elapsed = time.time() - start_time
        hours, remainder = divmod(
            datetime.timedelta(seconds=elapsed).total_seconds(), 3600
        )
        minutes, seconds = divmod(remainder, 60)
        logging.info(
            f"Loading problems took {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds."
        )
        return datasets

    def prepare_data(self) -> None:
        """
        This method needs to be called before fit/validation/etc.
        The datasets for all training / validation problems are loaded or newly constructed.
        If `parallel` was specified, the datasets will be loaded using multiprocessing.
        This will typically be slower if the datasets already exist and can be loaded directly.
        NOTE it is important for the validation problems to be in the same order as in
        :attr: `InputData.validation_problem_paths`!
        """
        if self._data_prepared:
            return

        train_prob_paths: List[Path] = self.data.problem_paths
        validation_prob_paths: List[Path] | None = self.data.validation_problem_paths
        problem_paths = train_prob_paths + (validation_prob_paths or [])
        logging.info(f"Using #{len(problem_paths)} problems in total.")
        logging.info(
            f"Problems used for TRAINING:\n"
            + "\n".join(p.stem for p in train_prob_paths)
        )
        validation_string = "-NONE-"
        if validation_prob_paths:
            validation_string = "\n".join(p.stem for p in validation_prob_paths)
        logging.info(f"Problems used for VALIDATION:\n{validation_string}")
        datasets: Dict[Path, Dataset] = self.load_datasets(problem_paths)
        train_desc = "\n".join(str(datasets[p]) for p in train_prob_paths)
        logging.info(f"Loaded TRAINING datasets:\n" f"{train_desc}")
        if validation_prob_paths:
            val_desc = "\n".join(str(datasets[p]) for p in validation_prob_paths)
            logging.info(f"Loaded VALIDATION datasets:\n" f"{val_desc}")

        self.dataset = ConcatDataset(
            [datasets[train_problem] for train_problem in train_prob_paths]
        )
        if validation_prob_paths:
            self.validation_sets = [
                datasets[val_problem] for val_problem in validation_prob_paths
            ]
        self._data_prepared = True
        if self.exit_after_processing:
            logging.info("Stopping after data processing desired. Exiting now.")
            exit(0)

    def _imbalanced_sampler(self):
        # We expect that every datapoint has a distance_to_goal attribute
        class_tensor = torch.cat(
            [dataset.distance_to_goal for dataset in self.dataset.datasets]
        )
        # Account for deadend state labels being -1 (not working with bincount usage inside `ImbalancedSampler`)
        # Adding +1 to each distance (label) doesnt change the relative class counts, so does not change the sampling
        class_tensor = class_tensor + 1
        return ImbalancedSampler(dataset=class_tensor)

    @property
    def train_datasets(self):
        if not self._data_prepared:
            self.prepare_data()
        return self.dataset.datasets

    @property
    def validation_datasets(self):
        if not self._data_prepared:
            self.prepare_data()
        return self.validation_sets

    @property
    def datasets(self):
        if not self._data_prepared:
            self.prepare_data()
        return self.train_datasets + self.validation_datasets

    def train_dataloader(self, **kwargs) -> TRAIN_DATALOADERS:
        defaults = dict(
            sampler=(
                self._imbalanced_sampler() if self.balance_by_distance_to_goal else None
            ),
            batch_size=self.batch_size,
            shuffle=not self.balance_by_distance_to_goal,
            num_workers=self.num_workers_train,
            persistent_workers=self.num_workers_train > 0,
        )
        defaults.update(kwargs)

        return DataLoader(
            self.dataset,
            collate_fn=collate_fn,
            **defaults,
        )

    def val_dataloader(self, **kwargs) -> TRAIN_DATALOADERS:
        # Order of dataloader has to be equal to order of validation problems in `InputData`.
        defaults = dict(
            batch_size=self.batch_size_validation,
            shuffle=False,
            num_workers=self.num_workers_val,
            # as we have multiple loader each individually should get fewer workers
            persistent_workers=False,  # when True validation memory usage increases a lot with every dataloader.
        )
        defaults.update(kwargs)
        return [
            DataLoader(
                dataset,
                collate_fn=collate_fn,
                **defaults,
            )
            for dataset in self.validation_sets
        ]
