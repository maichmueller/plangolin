from __future__ import annotations

import datetime
import logging
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch_geometric.loader import ImbalancedSampler

import rgnet.rl.thundeRL.collate as collate  # noqa: F401
from rgnet.encoding import GraphEncoderBase
from rgnet.rl.data import BaseDrive, FlashDrive
from rgnet.rl.data_layout import InputData
from rgnet.rl.envs import ExpandedStateSpaceEnv, LazyEnvLookup
from rgnet.rl.reward import RewardFunction
from rgnet.rl.thundeRL.collate import StatefulCollater
from rgnet.utils.misc import env_aware_cpu_count
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
        collate_fn: Callable = collate.to_transitions_batch,
        collate_kwargs: Optional[Dict[str, Any]] = None,
        drive_type: type[BaseDrive] = FlashDrive,
        drive_kwargs: Optional[Dict[str, Any]] = None,
        test_drive_type: type[BaseDrive] = None,
        test_drive_kwargs: Optional[Dict[str, Any]] = None,
        train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        validation_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        test_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        balance_by_attr: str = "",
        max_cpu_count: Optional[int] = None,
        skip: bool = False,
        exit_after_processing: bool = False,
    ) -> None:
        super().__init__()

        self._data_prepared = False
        self.data = input_data
        self.reward_function = reward_function
        self.batch_size = batch_size
        self.parallel = parallel
        self.drive_type = drive_type
        self.test_drive_type = test_drive_type or drive_type
        self.drive_kwargs = drive_kwargs or dict()
        self.test_drive_kwargs = test_drive_kwargs or dict()
        self.train_dataloader_kwargs = (
            dict(num_workers=6)
            | (train_dataloader_kwargs or dict())
            | dict(batch_size=batch_size)
        )  # do not override batch_size arg silently
        self.validation_dataloader_kwargs = dict(
            num_workers=2,
            batch_size=batch_size,
        ) | (validation_dataloader_kwargs or dict())
        self.test_dataloader_kwargs = dict(
            num_workers=2,
            batch_size=batch_size,
        ) | (test_dataloader_kwargs or dict())
        self.encoder_factory = encoder_factory
        self.balance_by_attr = balance_by_attr
        self.max_cpu_count = max_cpu_count
        self.exit_after_processing = exit_after_processing
        self.dataset: ConcatDataset | None = None  # late init in prepare_data()
        self.validation_sets: Sequence[Dataset] = []
        self.test_sets: Sequence[Dataset] = []
        self.collate_fn = collate_fn
        self.collate_kwargs = collate_kwargs or dict()
        self.envs: Mapping[Path, ExpandedStateSpaceEnv] = LazyEnvLookup(
            input_data.problem_paths + (input_data.validation_problem_paths or []),
            self.get_env,
        )
        # whether to skip the data preparation step completely (e.g. for testing)
        self.skip = skip
        # defaulted, to be overridden by trainer on setup
        self.device = torch.device("cpu")

    def _make_collate(self):
        kwargs = self.collate_kwargs.copy()
        from_datamodule = kwargs.pop("from_datamodule", None)
        if from_datamodule is not None:
            assert isinstance(from_datamodule, dict)
            for key, attr in from_datamodule.items():
                if isinstance(attr, str):
                    kwargs[key] = getattr(self, attr)
                elif isinstance(attr, list):
                    obj = self
                    for nested_attr in attr:
                        obj = getattr(obj, nested_attr)
                    kwargs[key] = obj
                else:
                    raise ValueError(
                        f"Invalid type for attribute '{key}': {type(attr)}. Expected str or list."
                    )
        return StatefulCollater(self.collate_fn, **kwargs)

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
        if (iw_search := self.drive_kwargs.get("iw_search")) is not None:
            return ExpandedStateSpaceEnv(
                IWStateSpace(
                    iw_search,
                    space,
                    **self.drive_kwargs.get("iw_options", dict()),
                ),
                reward_function=self.reward_function,
                reset=True,
            )
        else:
            return ExpandedStateSpaceEnv(space, reset=True)

    def load_datasets(
        self,
        problem_paths: Sequence[Path],
        drive_types: Sequence[Type[BaseDrive]] | None = None,
        drive_types_kwargs: Sequence[Dict[str, Any]] | None = None,
    ) -> Dict[Path, Dataset]:
        nr_total = len(problem_paths)
        completed = 0

        def update(dataset):
            nonlocal completed
            completed += 1
            logging.info(
                f"Finished loading {completed}/{nr_total} problems - Most recent loaded: {dataset.problem_path.stem} "
                f"(#{len(dataset)} states)."
            )

        datasets: Dict[Path, BaseDrive] = dict()
        drive_types = drive_types or [self.drive_type] * len(
            problem_paths
        )  # assume all the same drive type if missing
        drive_types_kwargs = drive_types_kwargs or [self.drive_kwargs] * len(
            problem_paths
        )  # assume all the same drive kwargs if missing
        assert len(drive_types) == len(problem_paths) == len(drive_types_kwargs)
        drive_extra_kwargs = dict(
            domain_path=self.data.domain_path,
            reward_function=self.reward_function,
            logging_kwargs=None,
            encoder_factory=self.encoder_factory,
        )
        start_time = time.time()
        if self.parallel and len(problem_paths) > 1:

            def enqueue_parallel(problem_path: Path, drive_t, drive_kw, task_id: int):
                return pool.apply_async(
                    drive_t,
                    kwds=drive_kw
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
                    problem_path: enqueue_parallel(
                        problem_path,
                        drive_t,
                        drive_kw | drive_extra_kwargs,
                        i,
                    )
                    for i, (problem_path, drive_t, drive_kw) in enumerate(
                        zip(problem_paths, drive_types, drive_types_kwargs)
                    )
                }
                for problem_path, result in results.items():
                    datasets[problem_path] = result.get()
        else:
            for problem_path, drive_t, drive_kw in zip(
                problem_paths, drive_types, drive_types_kwargs
            ):
                drive = drive_t(
                    problem_path=problem_path,
                    root_dir=str(self.data.dataset_dir / problem_path.stem),
                    show_progress=True,
                    env=self.envs(problem_path),
                    **(drive_kw | drive_extra_kwargs),
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
        if self._data_prepared or self.skip:
            return

        train_prob_paths: List[Path] = self.data.problem_paths
        validation_prob_paths: List[Path] = self.data.validation_problem_paths or []
        test_problem_paths: List[Path] = self.data.test_problem_paths or []
        problem_paths = train_prob_paths + validation_prob_paths + test_problem_paths
        logging.info(f"Using #{len(problem_paths)} problems in total.")
        logging.info(
            f"Problems used for TRAINING:\n"
            + "\n".join(p.stem for p in train_prob_paths)
        )
        validation_string = "-NONE-"
        if validation_prob_paths:
            validation_string = "\n".join(p.stem for p in validation_prob_paths)
        test_string = "-NONE-"
        if test_problem_paths:
            test_string = "\n".join(p.stem for p in test_problem_paths)

        logging.info(f"Problems used for VALIDATION:\n{validation_string}")
        logging.info(f"Problems used for TESTING:\n{test_string}")

        # the actual work intensive part of this function is loading the datasets
        datasets: Dict[Path, Dataset] = self.load_datasets(
            problem_paths,
            drive_types=[self.drive_type]
            * (len(train_prob_paths) + len(validation_prob_paths))
            + [self.test_drive_type] * len(test_problem_paths),
            drive_types_kwargs=[self.drive_kwargs]
            * (len(train_prob_paths) + len(validation_prob_paths))
            + [self.test_drive_kwargs] * len(test_problem_paths),
        )

        train_desc = "\n".join(str(datasets[p]) for p in train_prob_paths)
        logging.info(f"Loaded TRAINING datasets:\n" f"{train_desc}")
        if validation_prob_paths:
            val_desc = "\n".join(str(datasets[p]) for p in validation_prob_paths)
            logging.info(f"Loaded VALIDATION datasets:\n" f"{val_desc}")
        if test_problem_paths:
            logging.info(
                f"Loaded TEST datasets:\n"
                + "\n".join(
                    str(self.test_sets[p]) for p in self.data.test_problem_paths
                )
            )
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

    def _imbalanced_sampler(self) -> ImbalancedSampler | None:
        if self.balance_by_attr:
            class_tensor = torch.cat(
                [
                    getattr(dataset, self.balance_by_attr)
                    for dataset in self.train_datasets
                ]
            )
            # Account for deadend state labels being -1 (values <0 are incompatible with bincount inside `ImbalancedSampler`)
            # Adding the min to each distance (label) doesn't change the relative class counts, so does not change the sampling

            if (min_label := class_tensor.min()) < 0:
                class_tensor = class_tensor + (-min_label)
            return ImbalancedSampler(dataset=class_tensor)
        else:
            return None

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
    def test_datasets(self):
        if not self._data_prepared:
            self.prepare_data()
        return self.test_sets

    @property
    def datasets(self):
        if not self._data_prepared:
            self.prepare_data()
        return (
            self.train_datasets
            + list(self.validation_datasets)
            + list(self.test_datasets)
        )

    def _sanitize_dataloader_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # remove any key value combination that are not valid for DataLoader
        if (
            kwargs.get("persistent_workers", False)
            and kwargs.get("num_workers", 0) == 0
        ):
            # raises ValueError otherwise
            kwargs["persistent_workers"] = False
        return kwargs

    def train_dataloader(self, **kwargs) -> TRAIN_DATALOADERS:
        defaults = (
            dict(
                sampler=self._imbalanced_sampler(),
                batch_size=self.batch_size,
                shuffle=not self.balance_by_attr,
                collate_fn=self._make_collate(),
                pin_memory=True,
            )
            | self.train_dataloader_kwargs
        )
        return DataLoader(
            self.dataset,
            **self._sanitize_dataloader_kwargs(defaults | kwargs),
        )

    def val_dataloader(self, **kwargs) -> EVAL_DATALOADERS:
        # Order of dataloader has to be equal to order of validation problems in `InputData`.
        defaults = (
            dict(
                shuffle=False,
                collate_fn=self._make_collate(),
                pin_memory=True,
            )
            | self.validation_dataloader_kwargs
        )
        return [
            DataLoader(
                dataset,
                **self._sanitize_dataloader_kwargs(defaults | kwargs),
            )
            for dataset in self.validation_sets
        ]

    def test_dataloader(self, **kwargs) -> EVAL_DATALOADERS:
        # Order of dataloader should be equal to order of test problems in `InputData`.
        defaults = (
            dict(
                shuffle=False,
                collate_fn=self._make_collate(),
                pin_memory=True,
            )
            | self.test_dataloader_kwargs
        )
        return [
            DataLoader(
                dataset,
                **self._sanitize_dataloader_kwargs(defaults | kwargs),
            )
            for dataset in self.test_sets
        ]
