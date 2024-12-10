import logging
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch_geometric.loader import ImbalancedSampler

from rgnet.encoding import GraphEncoderBase
from rgnet.rl.data_layout import InputData
from rgnet.rl.thundeRL.collate import collate_fn
from rgnet.rl.thundeRL.flash_drive import FlashDrive


class ThundeRLDataModule(LightningDataModule):
    def __init__(
        self,
        input_data: InputData,
        gamma: float,
        batch_size: int,
        encoder_type: Type[GraphEncoderBase],
        *,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        balance_by_distance_to_goal: bool = True,
    ) -> None:
        super().__init__()

        self.data = input_data
        self.gamma = gamma
        self.batch_size = batch_size
        self.parallel = parallel
        self.encoder_type = encoder_type
        self.encoder_kwargs = encoder_kwargs or dict()
        self.balance_by_distance_to_goal = balance_by_distance_to_goal
        self.dataset: ConcatDataset | None = None  # late init in prepare_data()
        self.validation_sets: List[Dataset] = []

    def load_datasets(self, problem_paths: List[Path]) -> Dict[Path, Dataset]:

        def update(dataset):
            logging.info(
                f"Finished loading problem {dataset.problem_path.stem} (#{len(dataset)} states)."
            )

        datasets: Dict[Path, FlashDrive] = dict()
        flashdrive_kwargs = dict(
            domain_path=self.data.domain_path,
            custom_dead_end_reward=-1 / (1 - self.gamma),
            root_dir=str(self.data.dataset_dir),
            logging_kwargs=None,
            encoder_type=self.encoder_type,
            encoder_kwargs=self.encoder_kwargs,
        )
        if self.parallel and len(problem_paths) > 1:

            def enqueue_parallel(problem_path: Path, thread_id: int):
                return pool.apply_async(
                    FlashDrive,
                    kwds=flashdrive_kwargs
                    | dict(
                        problem_path=problem_path,
                        show_progress=False,
                        logging_kwargs=dict(
                            log_level=logging.getLogger().level, thread_id=thread_id
                        ),
                    ),
                    callback=update,
                )

            with Pool(min(cpu_count(), len(problem_paths))) as pool:
                logging.info(f"Loading #{len(problem_paths)} problems in parallel.")
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
                    show_progress=True,
                    **flashdrive_kwargs,
                )
                update(drive)
                datasets[problem_path] = drive
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
        if self.dataset is not None:
            warnings.warn(
                "Called prepare_data() but the data is already loaded."
                "Replacing datasets ..."
            )

        train_prob_paths: List[Path] = self.data.problem_paths
        validation_prob_paths: List[Path] | None = self.data.validation_problem_paths
        problem_paths = train_prob_paths + (validation_prob_paths or [])
        logging.info(f"Using #{len(problem_paths)} problems in total.")
        logging.info(
            f"Problems used for TRAINING:\n"
            + "\n".join(p.stem for p in self.data.problem_paths)
        )
        validation_string = "-NONE-"
        if validation_prob_paths:
            validation_string = "\n".join(p.stem for p in validation_prob_paths)
        logging.info(f"Problems used for VALIDATION:\n{validation_string}")
        datasets: Dict[Path, Dataset] = self.load_datasets(problem_paths)
        self.dataset = ConcatDataset(
            [datasets[train_problem] for train_problem in train_prob_paths]
        )
        if validation_prob_paths:
            self.validation_sets = [
                datasets[val_problem] for val_problem in validation_prob_paths
            ]

    def _imbalanced_sampler(self):
        # We expect that every datapoint has a distance_to_goal attribute
        class_tensor = torch.cat(
            [dataset.distance_to_goal for dataset in self.dataset.datasets]
        )
        return ImbalancedSampler(dataset=class_tensor)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset,
            sampler=(
                self._imbalanced_sampler() if self.balance_by_distance_to_goal else None
            ),
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=not self.balance_by_distance_to_goal,
            num_workers=6,
            persistent_workers=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        # Order of dataloader has to be equal to order of validation problems in `InputData`.
        return [
            DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                # as we have multiple loader each individually should get less worker
                persistent_workers=True,
            )
            for dataset in self.validation_sets
        ]
