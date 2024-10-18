import logging
from pathlib import Path
from typing import List

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from experiments.rl.data_layout import InputData
from rgnet.rl.thundeRL.collate import collate_fn
from rgnet.rl.thundeRL.flash_drive import FlashDrive


class ThundeRLDataModule(LightningDataModule):

    def __init__(
        self,
        input_data: InputData,
        gamma: float,
        batch_size: int,
    ) -> None:
        super().__init__()

        self.data = input_data

        self.gamma = gamma
        self.batch_size = batch_size
        self.dataset: Dataset
        self.validation_sets: List[Dataset] = []

    def load_datasets(self, problem_paths: List[Path]):
        if len(problem_paths) > 3:
            logging.info(f"Using {len(problem_paths)} problems for training")
        else:
            join = "\n".join([p.stem for p in problem_paths])
            logging.info(f"Using problems: {join}")
        dataset_list = []
        for problem_path in problem_paths:
            dataset_list.append(
                FlashDrive(
                    domain_path=self.data.domain_path,
                    problem_path=problem_path,
                    custom_dead_end_reward=-1 / (1 - self.gamma),
                    root_dir=str(self.data.dataset_dir),
                )
            )
        return dataset_list

    def prepare_data(self) -> None:
        self.dataset = ConcatDataset(self.load_datasets(self.data.problem_paths))
        if self.data.validation_problems:
            self.validation_sets = self.load_datasets(
                self.data.validation_problem_paths
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return [
            DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
                persistent_workers=True,
            )
            for dataset in self.validation_sets
        ]
