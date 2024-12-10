import argparse
import os
import time
from datetime import datetime

import lightning.pytorch.callbacks
import torch_geometric as pyg
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import ImbalancedSampler

from experiments import PlanResult
from experiments.analyze_run import CompletedExperiment
from experiments.data_layout import DataLayout, DatasetType
from rgnet.encoding import (
    ColorGraphEncoder,
    DirectGraphEncoder,
    GraphEncoderBase,
    HeteroGraphEncoder,
)
from rgnet.models import LightningHetero, VanillaGNN
from rgnet.supervised.over_sampler import OverSampler
from rgnet.supervised.parse_serialized_dataset import *
from rgnet.utils import import_problems, time_delta_now


def load_serialized(
    domain: mi.Domain,
    encoder: GraphEncoderBase,
    dataset_type: DatasetType,
    data_layout: DataLayout,
):
    problems = list(data_layout.problems_paths_for(dataset_type))
    serialized_paths: List[Path] = list(data_layout.serialized_paths_for(dataset_type))
    prob_to_serialized: Dict[Path, Path] = dict()
    for s_path in serialized_paths:
        try:
            prob_file = data_layout.problem_for_serialized(s_path, dataset_type)
            prob_to_serialized[prob_file] = s_path
        except ValueError as e:  # There was no problem file for this serialized file.
            logging.warning(repr(e))
    if len(problems) != len(prob_to_serialized):
        logging.warning(f"Mismatch between problems and serialized files.")

    prob_to_serialized: Dict[mi.Problem, Path] = {
        mi.ProblemParser(str(problem_path)).parse(domain): serialized
        for problem_path, serialized in prob_to_serialized.items()
    }
    dataset_path = data_layout.dataset_path_for(dataset_type)
    logging.info("Creating serialized dataset: " + str(dataset_path.absolute()))
    dataset = SerializedDataset(
        domain,
        prob_to_serialized,
        problems=prob_to_serialized.keys(),
        state_encoder=encoder,
        root=dataset_path,
        log=True,
    )
    return dataset


def _dataset_of(problems, root, encoder: GraphEncoderBase):
    return MultiInstanceSupervisedSet(problems, encoder, root=root, log=True)


def _setup_datasets(
    data_layout: DataLayout,
    batch_size: int,
    num_samples: int,
    oversampling_factor: float | None,
):
    domain = mi.DomainParser(str(data_layout.domain_file_path.absolute())).parse()

    encoder = _create_encoder(domain, data_layout.encoder_type)
    loaders = []
    logging.info(f"Loading/Building dataset at {data_layout.dataset_path}")
    dataset_type: DatasetType
    for dataset_type in DatasetType:
        serialized_paths = list(data_layout.serialized_paths_for(dataset_type))
        if len(serialized_paths) > 0:
            dataset = load_serialized(domain, encoder, dataset_type, data_layout)
        else:
            problem_path = data_layout.problems_path_for(dataset_type)
            problems = import_problems(problem_path, domain)
            dataset_path = data_layout.dataset_path_for(dataset_type)
            dataset = _dataset_of(problems, dataset_path, encoder)
        if dataset_type == DatasetType.TRAIN and oversampling_factor is not None:
            sampler = OverSampler(
                dataset,
                oversampled_class=0,
                oversampling_factor=oversampling_factor,
                num_samples=num_samples,
            )
        else:
            sampler = ImbalancedSampler(dataset.y, num_samples=num_samples)
        logging.info(
            f"Using {type(sampler)} sampler with {num_samples} samples for {dataset_type}"
        )
        loader = pyg.loader.DataLoader(
            dataset, batch_size, shuffle=False, sampler=sampler, num_workers=2
        )
        loader = pyg.loader.PrefetchLoader(loader)
        logging.info(
            f"Dataset for {dataset_type} contains {len(dataset)} graphs/states"
        )
        loaders.append(loader)

    train_loader, eval_loader, test_loader = loaders

    return train_loader, eval_loader, test_loader, encoder


def _create_encoder(domain: mi.Domain, encoder_type: str) -> GraphEncoderBase:
    if encoder_type == "color":
        encoder = ColorGraphEncoder(domain)
    elif encoder_type == "direct":
        encoder = DirectGraphEncoder(domain)
    elif encoder_type == "hetero":
        encoder = HeteroGraphEncoder(domain)
    else:
        raise ValueError(f"Encoding type {encoder_type} not recognized.")
    return encoder


def plan(data_layout, run_id, logger: WandbLogger):
    exp = CompletedExperiment(data_layout, run_id)
    results: List[PlanResult] = exp.run_vpolicy(DatasetType.TEST)
    # add results to wandb as table
    data = []
    for result in results:
        data.append(
            [result.problem, result.cost, result.opt_cost],
        )
    logger.log_table(
        "plans", columns=["problem", "plan cost", "optimal cost"], data=data
    )
    opt_solved = sum(1 for r in results if r.plan and r.cost == r.opt_cost)
    subopt_solved = sum(1 for r in results if r.plan and r.cost > r.opt_cost)
    unsolved = sum(1 for r in results if not r.plan)
    logger.log_metrics(
        {
            "optimal solved": opt_solved,
            "suboptimal solved": subopt_solved,
            "unsolved": unsolved,
        }
    )


def run(
    encoder_type,
    domain_name,
    batch_size,
    data_path,
    device,
    epochs,
    num_samples,
    oversampling_factor,
    **kwargs,  # model args
):
    # Fixes RuntimeError: received 0 items of ancdata which occurs with some domains
    # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
    torch.multiprocessing.set_sharing_strategy("file_system")
    curr_dir = os.getcwd()
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Using {device} as device")
    logging.info("Working from " + curr_dir)
    start_time = time.time()
    # day-month-year_hour-minute-second
    time_stamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    wlogger = WandbLogger(
        project="rgnet",
        name=f"{domain_name}:{time_stamp}",
        # group="reproducibility",
    )
    run_id = wlogger.experiment.id
    # Model hyperparameter saved by Lightning
    wlogger.experiment.config.update(
        {
            "epoch": epochs,
            "device": device,
            "encoding": encoder_type,
            "domain": domain_name,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "oversampling_factor": oversampling_factor,
        },
    )

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    data_layout = DataLayout(data_path, domain_name, encoder_type)
    import_time = time.time()

    train_loader, eval_loader, test_loader, encoder = _setup_datasets(
        data_layout, batch_size, num_samples, oversampling_factor
    )

    logging.info(f"Took {time_delta_now(import_time)} to construct the datasets.")

    start_time_training = time.time()
    if isinstance(encoder, HeteroGraphEncoder):
        model = LightningHetero(
            **kwargs,
            obj_type_id=encoder.obj_type_id,
            arity_dict=encoder.arity_dict,
        )
        wlogger.watch(model.model)
    else:
        model = VanillaGNN(
            size_out=1,
            size_in=1,
            hidden_size=kwargs["hidden_size"],
            num_layer=kwargs["num_layer"],
        )
        wlogger.watch(model)

    wlogger.experiment.config["num_parameter"] = model.num_parameter()

    checkpoint_path = data_layout.model_save_path / run_id
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Lightning prioritizes the logger-save dir over its own default_dir for
    # _checkpoints. Therefore, we have to use this callback (which has top priority).
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path
    )
    trainer = Trainer(
        accelerator=device,
        devices=1,
        max_epochs=epochs,
        logger=wlogger,
        callbacks=[checkpoint_callback],
    )

    logging.info("Starting training")
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=eval_loader
    )
    logging.info(f"Took {time_delta_now(start_time_training)} to train the model")

    trainer.test(model, dataloaders=test_loader)

    logging.info(f"Completed run after {time_delta_now(start_time)}.")
    torch.save(model.state_dict(), checkpoint_path / "model.pt")
    logging.info("Saved model to " + str(checkpoint_path))

    plan_start_time = time.time()
    logging.info(f"Starting plan for {run_id}")
    plan(data_layout, run_id, wlogger)
    logging.info(f"Finished plan for {run_id} in {time_delta_now(plan_start_time)}")

    wandb.finish()


if __name__ == "__main__":
    # Define default args
    DEFAULT_ENCODING = "hetero"
    DEFAULT_DOMAIN = "blocks-on"
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_DATA_PATH = "../data"
    DEFAULT_DEVICES = "auto"
    DEFAULT_EPOCHS = 30

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some modules.")
    parser = LightningHetero.add_model_args(parser)

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
        "--num_samples",
        type=int,
        default=None,
        help=f"Maximum number of samples used per epoch (default: {None})",
    )
    parser.add_argument(
        "--oversampling_factor",
        type=float,
        default=None,
        help=f"Oversample goals by this factor (default: {None})",
    )

    # Parse the arguments
    args = parser.parse_args()
    run(**vars(args))
