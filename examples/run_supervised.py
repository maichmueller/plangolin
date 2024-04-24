import argparse
import logging
import os
import pathlib
import time
from datetime import datetime

import lightning
import torch
import torch_geometric as pyg
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from rgnet.encoding import ColorGraphEncoder, DirectGraphEncoder, StateGraphEncoderBase
from rgnet.model import PureGNN
from rgnet.supervised.data import MultiInstanceSupervisedSet
from rgnet.utils import (
    get_device_cuda_if_possible,
    import_all_from,
    import_problems,
    time_delta_now,
)


def _dataset_of(problems, root, encoder: StateGraphEncoderBase):
    return MultiInstanceSupervisedSet(problems, encoder, root=root, log=True)


def _setup_datasets(
    encoding: str, problem_path: str, dataset_path: str, batch_size: int
):
    domain, problems = import_all_from(problem_path)
    if encoding == "color":
        encoder = ColorGraphEncoder(domain)
    elif encoding == "direct":
        encoder = DirectGraphEncoder(domain)
    else:
        raise ValueError(f"Encoding type {encoding} not recognized.")
    training_set = _dataset_of(problems, dataset_path + "/train", encoder)
    train_loader = pyg.loader.DataLoader(
        training_set, batch_size, shuffle=True, num_workers=10
    )

    evaluation_set = _dataset_of(
        import_problems(problem_path + "/eval", domain), dataset_path + "/eval", encoder
    )
    # Test the model
    test_set = _dataset_of(
        import_problems(problem_path + "/test", domain), dataset_path + "/test", encoder
    )
    eval_loader, test_loader = [
        pyg.loader.DataLoader(dset, batch_size, shuffle=False, num_workers=10)
        for dset in (evaluation_set, test_set)
    ]

    logging.info(f"Training dataset contains {len(training_set)} graphs/states")
    logging.info(f"Evaluation dataset contains {len(evaluation_set)} graphs/states")

    return train_loader, eval_loader, test_loader


def run(data_path, epochs, embedding_size, num_layer, batch_size, device, encoding):
    curr_dir = os.getcwd()
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Using {device} as device")
    logging.info("Working from " + curr_dir)

    start_time = time.time()
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    wlogger = WandbLogger(project="rgnet", name="supervised_blocks" + time_stamp)
    wlogger.experiment.config.update(
        {
            "epoch": epochs,
            "embedding_size": embedding_size,
            "num_layer": num_layer,
            "device": device,
        },
    )

    seed_everything(42, workers=True)

    # define paths
    problem_path = data_path + "/pddl_domains/blocks"
    dataset_path = data_path + "/datasets/blocks"
    model_save_path = pathlib.Path(data_path + f"/models/run{time_stamp}.pt")
    model_save_path.parent.mkdir(exist_ok=True)

    import_time = time.time()

    train_loader, eval_loader, test_loader = _setup_datasets(
        encoding, problem_path, dataset_path, batch_size
    )

    logging.info(f"Took {time_delta_now(import_time)} to construct the datasets.")

    start_time_training = time.time()

    model = PureGNN(
        size_out=1, size_in=1, size_embedding=embedding_size, num_layer=num_layer
    )
    wlogger.watch(model)
    wlogger.experiment.config.update({"num_parameter": model.num_parameter()})

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        logger=wlogger,
    )

    logging.info("Starting training")
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=eval_loader
    )
    logging.info(f"Took {time_delta_now(start_time_training)} to train the model")

    logging.info(f"Saved model can be found at {model_save_path}")

    trainer.test(model, dataloaders=test_loader)

    logging.info(f"Completed run after {time_delta_now(start_time)}.")

    wandb.finish(quiet=True)


if __name__ == "__main__":
    # Define capitalized default variables
    DEFAULT_ENCODING = "color"
    DEFAULT_EMBEDDING_SIZE = 32
    DEFAULT_NUM_LAYER = 24
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_DATA_PATH = "./data"
    DEFAULT_DEVICES = "auto"
    DEFAULT_EPOCHS = 10

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some modules.")

    # Add arguments
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        choices=["color", "direct"],
        help=f"Encoding type (default: {DEFAULT_ENCODING})",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=DEFAULT_EMBEDDING_SIZE,
        help=f"Embedding size (default: {DEFAULT_EMBEDDING_SIZE})",
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=DEFAULT_NUM_LAYER,
        help=f"Number of layers (default: {DEFAULT_NUM_LAYER})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to data directory (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICES,
        help=f"Device to use (default: {DEFAULT_DEVICES})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})",
    )

    # add an arg for `accelerator`:

    # Parse the arguments
    args = parser.parse_args()
    run(**vars(args))
