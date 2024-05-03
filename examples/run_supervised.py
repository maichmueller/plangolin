import argparse
import os
import pathlib
import time
from datetime import datetime

import torch_geometric as pyg
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import ImbalancedSampler

from rgnet import (
    ColorGraphEncoder,
    DirectGraphEncoder,
    HeteroGraphEncoder,
    LightningHetero,
    PureGNN,
    StateEncoderBase,
)
from rgnet.supervised.parse_serialized_dataset import *
from rgnet.utils import import_problems, time_delta_now


def load_serialized(
    domain: mi.Domain,
    encoder: StateEncoderBase,
    problem_path: str,
    dataset_path: str,
    serialized_path: str,
):
    lookup = match_problems(problem_path, serialized_path)
    lookup = {
        mi.ProblemParser(str(problem_path)).parse(domain): serialized
        for problem_path, serialized in lookup.items()
    }
    logging.info("Creating serialized dataset: " + dataset_path)
    dataset = SerializedDataset(
        domain,
        lookup,
        problems=lookup.keys(),
        state_encoder=encoder,
        root=dataset_path,
        log=True,
        force_reload=True,
    )
    return dataset


def _dataset_of(problems, root, encoder: StateEncoderBase):
    return MultiInstanceSupervisedSet(problems, encoder, root=root, log=True)


def _setup_datasets(
    encoder_type: str,
    problem_path: str,
    dataset_path: str,
    batch_size: int,
    serialized_path: str | None = None,
):
    domain = mi.DomainParser(problem_path + "/domain.pddl").parse()

    encoder = _create_encoder(domain, encoder_type)
    loaders = []
    logging.info(f"Loading/Building dataset at {dataset_path}")
    for mode in ["train", "eval", "test"]:
        if serialized_path:
            dataset = load_serialized(
                domain,
                encoder,
                f"{problem_path}/{mode}",
                f"{dataset_path}/{mode}",
                f"{serialized_path}/{mode}",
            )
        else:
            problems = import_problems(f"{problem_path}/{mode}", domain)
            dataset = _dataset_of(problems, f"{dataset_path}/{mode}", encoder)
        sampler = ImbalancedSampler(dataset)
        loader = pyg.loader.DataLoader(
            dataset, batch_size, shuffle=False, sampler=sampler, num_workers=2
        )
        logging.info(f"Dataset for {mode} contains {len(dataset)} graphs/states")
        loaders.append(loader)

    train_loader, eval_loader, test_loader = loaders

    return train_loader, eval_loader, test_loader, encoder


def _create_encoder(domain: mi.Domain, encoder_type: str) -> StateEncoderBase:
    if encoder_type == "color":
        encoder = ColorGraphEncoder(domain)
    elif encoder_type == "direct":
        encoder = DirectGraphEncoder(domain)
    elif encoder_type == "hetero":
        encoder = HeteroGraphEncoder(domain)
    else:
        raise ValueError(f"Encoding type {encoder_type} not recognized.")
    return encoder


def run(
    data_path,
    domain_name,
    epochs,
    embedding_size,
    num_layer,
    batch_size,
    device,
    encoder_type,
    learning_rate,
):
    curr_dir = os.getcwd()
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Using {device} as device")
    logging.info("Working from " + curr_dir)
    start_time = time.time()
    # day-month-year_hour-minute-second
    time_stamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    wlogger = WandbLogger(
        project="rgnet", name=f"{encoder_type}/{domain_name}:{time_stamp}"
    )
    wlogger.experiment.config.update(
        {
            "epoch": epochs,
            "embedding_size": embedding_size,
            "num_layer": num_layer,
            "learning_rate": learning_rate,
            "device": device,
            "encoding": encoder_type,
            "domain": domain_name,
            "batch_size": batch_size,
        },
    )

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    """
    Folder structure:

    data_path
        - pddl_domains
            - <domain-name>
                - domain.pddl
                - train
                    - problem1.pddl
                    - ...
                - eval
                - test
        - serialized
            - <domain-name>
                - train
                    - problem1_states.txt
                    - ...
                - eval
                - test
        - datasets
            - <encoding_type>
                - <domain-name>
                    - train #  root dir of dataset
                        - processed
                        - raw
                    - eval
                    - test
        - models
            - <encoding_type>
                - <domain-name>
                    - checkpoint.ckpt
    """
    if data_path[-1] == "/":  # remove training slash
        data_path = data_path[:-1]

    # define paths
    if not pathlib.Path(data_path).is_dir():
        logging.error(f"Data path {data_path} does not exist or is not a directory.")
        exit("Data path does not exist or is not a directory.")
    problem_path = f"{data_path}/pddl_domains/{domain_name}"
    dataset_path = f"{data_path}/datasets/{encoder_type}/{domain_name}"
    serialized_path = f"{data_path}/serialized/{domain_name}"
    # Create folder structure for datasets if absent
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    model_save_path = pathlib.Path(
        f"{data_path}/models/{encoder_type}/{domain_name}/run{time_stamp}/"
    )
    model_save_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using model-save-path: {model_save_path.absolute()}")

    import_time = time.time()

    train_loader, eval_loader, test_loader, encoder = _setup_datasets(
        encoder_type,
        problem_path,
        dataset_path,
        batch_size,
        serialized_path=serialized_path,
    )

    logging.info(f"Took {time_delta_now(import_time)} to construct the datasets.")

    start_time_training = time.time()
    if isinstance(encoder, HeteroGraphEncoder):
        model = LightningHetero(
            hidden_size=embedding_size,
            num_layer=num_layer,
            obj_type_id=encoder.obj_type_id,
            arity_dict=encoder.arity_dict,
            lr=learning_rate,
        )
        wlogger.watch(model.model)
    else:
        model = PureGNN(
            size_out=1, size_in=1, size_embedding=embedding_size, num_layer=num_layer
        )
        wlogger.watch(model)

    wlogger.experiment.config.update({"num_parameter": model.num_parameter()})

    trainer = Trainer(
        accelerator=device,
        devices=1,
        max_epochs=epochs,
        logger=wlogger,
        default_root_dir=model_save_path,
    )

    logging.info("Starting training")
    trainer.fit(model=model, train_dataloaders=eval_loader, val_dataloaders=eval_loader)
    logging.info(f"Took {time_delta_now(start_time_training)} to train the model")

    trainer.test(model, dataloaders=test_loader)

    logging.info(f"Completed run after {time_delta_now(start_time)}.")
    torch.save(model.state_dict(), model_save_path / "model.pt")

    wandb.finish(quiet=True)


if __name__ == "__main__":
    # Define default args
    DEFAULT_ENCODING = "color"
    DEFAULT_DOMAIN = "blocks"
    DEFAULT_EMBEDDING_SIZE = 32
    DEFAULT_NUM_LAYER = 24
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_DATA_PATH = "./data"
    DEFAULT_DEVICES = "auto"
    DEFAULT_EPOCHS = 10
    DEFAULT_LEARNING_RATE = 0.001

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some modules.")

    # Add arguments
    parser.add_argument(
        "--encoding",
        dest="encoder_type",
        default=DEFAULT_ENCODING,
        choices=["color", "direct", "hetero"],
        help=f"Encoding type (default: {DEFAULT_ENCODING})",
    )
    parser.add_argument(
        "--domain",
        dest="domain_name",
        default=DEFAULT_DOMAIN,
        help=f"Encoding type (default: {DEFAULT_DOMAIN})",
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
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate used by Adam (default: {DEFAULT_LEARNING_RATE})",
    )

    # Parse the arguments
    args = parser.parse_args()
    run(**vars(args))
