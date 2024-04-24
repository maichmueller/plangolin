import logging
import os
import pathlib
import time
from datetime import datetime

import torch
import torch_geometric as pyg
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import ImbalancedSampler

from rgnet.encoding.hetero import HeteroEncoding
from rgnet.models.hetero_gnn import LightningHetero
from rgnet.pddl_import import import_all_from, import_problems
from rgnet.supervised.data import MultiInstanceSupervisedSet
from rgnet.utils import get_device_cuda_if_possible, time_delta_now


def _dataset_of(problems, encoder, root):
    return MultiInstanceSupervisedSet(problems, encoder, root=root, log=True)


def _setup_datasets(problem_path: str, dataset_path: str, batch_size: int, hidden: int):
    domain, problems = import_all_from(problem_path)
    problems = sorted(problems, key=lambda p: p.name)
    encoder = HeteroEncoding(domain, hidden)
    training_set = _dataset_of(problems, encoder, dataset_path + "/train")
    sampler = ImbalancedSampler(training_set)
    train_loader = pyg.loader.DataLoader(
        training_set, batch_size, shuffle=False, sampler=sampler
    )

    evaluation_set = _dataset_of(
        import_problems(problem_path + "/eval", domain), encoder, dataset_path + "/eval"
    )
    # Test the model
    eval_sampler = ImbalancedSampler(evaluation_set)
    eval_loader = pyg.loader.DataLoader(
        evaluation_set, batch_size, shuffle=False, sampler=eval_sampler
    )

    logging.info(f"Training dataset contains {len(training_set)} graphs/states")
    logging.info(f"Evaluation dataset contains {len(evaluation_set)} graphs/states")

    return train_loader, eval_loader, encoder


def run(
    epochs=1,
    embedding_size=32,
    num_layer=24,
    batch_size=32,
    learning_rate=0.001,
):
    curr_dir = os.getcwd()
    logging.getLogger().setLevel(logging.INFO)
    device: torch.device = get_device_cuda_if_possible()
    logging.info(f"Using {device.type} as device")
    logging.info("Working from " + curr_dir)
    start_time = time.time()
    time_stamp = datetime.now().strftime("%y:%m:%d_%H:%M:%S")
    wlogger = WandbLogger(project="rgnet", name="heter_ImbalancedSampler" + time_stamp)
    wlogger.experiment.config.update(
        {
            "epoch": epochs,
            "embedding_size": embedding_size,
            "num_layer": num_layer,
            "learning_rate": learning_rate,
            "device": device.type,
        },
    )

    seed_everything(42, workers=True)

    # define paths
    data_path = curr_dir + "/data"
    problem_path = data_path + "/pddl_domains/blocks"
    dataset_path = data_path + "/datasets/blocks"
    model_save_path = pathlib.Path(data_path + f"/models/run{time_stamp}.pt")
    model_save_path.parent.mkdir(exist_ok=True)

    import_time = time.time()

    train_loader, eval_loader, encoder = _setup_datasets(
        problem_path, dataset_path, batch_size, hidden=embedding_size
    )

    logging.info(f"Took {time_delta_now(import_time)} to construct the datasets.")

    start_time_training = time.time()
    arity_by_pred = encoder.arity_by_pred
    model = LightningHetero(
        hidden_size=embedding_size,
        num_layer=num_layer,
        obj_name=encoder.obj_name,
        arity_by_pred=arity_by_pred,
        lr=learning_rate,
    )
    # wlogger.watch(model.model)
    # wlogger.experiment.config.update({"num_parameter": model.num_parameter()})

    trainer = Trainer(accelerator="auto", devices=1, max_epochs=epochs, logger=wlogger)

    logging.info("Starting training")
    trainer.fit(model=model, train_dataloaders=eval_loader, val_dataloaders=eval_loader)
    logging.info(f"Took {time_delta_now(start_time_training)} to train the model")

    logging.info(f"Saved model can be found at {model_save_path}")

    # trainer.test(model, dataloaders=test_loader)

    logging.info(f"Completed run after {time_delta_now(start_time)}.")

    wandb.finish(quiet=True)


if __name__ == "__main__":
    run()
