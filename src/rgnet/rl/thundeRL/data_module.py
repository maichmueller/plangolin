import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from experiments.rl.data_layout import InputData
from rgnet.rl.thundeRL.collate import collate_fn
from rgnet.rl.thundeRL.flash_drive import FlashDrive

_newline = "\n"


class ThundeRLDataModule(LightningDataModule):
    def __init__(
        self,
        input_data: InputData,
        gamma: float,
        batch_size: int,
        parallel: bool = True,
    ) -> None:
        super().__init__()

        self.data = input_data
        self.gamma = gamma
        self.batch_size = batch_size
        self.parallel = parallel
        self.dataset: Dataset
        self.validation_sets: List[Dataset] = []

    def load_datasets(self, problem_paths: List[Path]) -> List[Dataset]:

        def update(dataset):
            logging.info(
                f"Finished loading problem {dataset.problem_path.stem} (#{len(dataset)} states)."
            )

        process_parallely = self.parallel and len(problem_paths) > 1
        dataset_list = []
        flashdrive_kwargs = dict(
            domain_path=self.data.domain_path,
            custom_dead_end_reward=-1 / (1 - self.gamma),
            root_dir=str(self.data.dataset_dir),
            logging_kwargs=None,
        )
        if process_parallely:
            with Pool(min(cpu_count(), len(problem_paths))) as pool:
                logging.info(f"Loading #{len(problem_paths)} problems in parallel.")
                results = [
                    pool.apply_async(
                        FlashDrive,
                        kwds=flashdrive_kwargs
                        | dict(
                            problem_path=problem_path,
                            show_progress=False,
                            logging_kwargs=dict(
                                log_level=logging.getLogger().level, thread_id=i
                            ),
                        ),
                        callback=update,
                    )
                    for i, problem_path in enumerate(problem_paths)
                ]
                for result in results:
                    drive = result.get()
                    dataset_list.append(drive)
        else:
            for problem_path in problem_paths:
                dataset_list.append(
                    FlashDrive(
                        problem_path=problem_path,
                        show_progress=True,
                        **flashdrive_kwargs,
                    )
                )
                update(dataset_list[-1])
        return dataset_list

    def prepare_data(self) -> None:
        problem_paths = self.data.problem_paths + self.data.validation_problem_paths
        logging.info(f"Using #{len(problem_paths)} problems in total.")
        logging.info(
            f"Problems used for TRAINING:\n{_newline.join(p.stem for p in self.data.problem_paths)}"
        )
        logging.info(
            f"Problems used for VALIDATION:\n{_newline.join(p.stem for p in self.data.validation_problem_paths)}"
        )
        datasets = self.load_datasets(problem_paths)
        self.dataset = ConcatDataset(
            filter(
                lambda drive: drive.problem_path in self.data.problem_paths, datasets
            )
        )
        if self.data.validation_problems:
            self.validation_sets = list(
                filter(
                    lambda drive: drive.problem_path
                    in self.data.validation_problem_paths,
                    datasets,
                )
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            persistent_workers=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return [
            DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=6,
                persistent_workers=True,
            )
            for dataset in self.validation_sets
        ]
