import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import lightning
import torch
import torch.nn
from lightning.pytorch.loggers import WandbLogger

from experiments.rl.configs import agent as agent_module
from experiments.rl.configs import data_resolver
from experiments.rl.configs import embedding as embedding_module
from experiments.rl.configs import environment
from experiments.rl.configs import logger as logger_module
from experiments.rl.configs import trainer as trainer_module
from experiments.rl.configs import value_estimator
from experiments.rl.configs.agent import Agent
from experiments.rl.configs.trainer import Parameter as TrainerParameter
from experiments.rl.data_resolver import DataResolver
from rgnet import HeteroGNN
from rgnet.rl import ActorCritic, EmbeddingModule
from rgnet.rl.losses import CriticLoss
from rgnet.rl.thundeRL.collate import collate_fn
from rgnet.rl.thundeRL.flash_drive import FlashDrive
from rgnet.rl.thundeRL.lightning_adapter import LightningAdapter


def create_args():
    parser = ArgumentParser()
    parser = agent_module.add_parser_args(parser)
    parser = environment.add_parser_args(parser)
    parser = embedding_module.add_parser_args(parser)
    parser = logger_module.add_parser_args(parser)
    parser = trainer_module.add_parser_args(parser)
    parser = value_estimator.add_parser_args(parser)
    parser = data_resolver.add_parser_args(parser)
    parser.add_argument(
        "--gamma",
        type=float,
        required=False,
        default=0.9,
        help="Discount factor for the environment (default: 0.9)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        required=False,
        default="auto",
        help="Device to use (default: cuda if available else cpu)",
    )

    return parser


def _resolve_dataset(data_resolver: DataResolver, gamma: float):
    root_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "flash_drives"
        / data_resolver.domain.name
        / "train"
    )
    if len(data_resolver.problem_paths) > 3:
        logging.info(f"Using {len(data_resolver.problem_paths)} problems for training")
    else:
        join = "\n".join([p.stem for p in data_resolver.problem_paths])
        logging.info(f"Using problems: {join}")
    driver_list = []
    for problem_path in data_resolver.problem_paths:
        driver = FlashDrive(
            root_dir=str(root_dir),
            domain_path=data_resolver.domain_path,
            problem_path=problem_path,
            custom_dead_enc_reward=1.0 / (1.0 - gamma),
        )
        driver_list.append(driver)

    complete_dataset = torch.utils.data.ConcatDataset(driver_list)
    logging.info(f"Total of {len(complete_dataset)} training data points")
    return complete_dataset


def train(
    parser_args,
    data: DataResolver,
    agent: ActorCritic,
    gnn: HeteroGNN,
    batch_size: int,
    device: str,
    epochs: int,
    loss: CriticLoss,
    optim: torch.optim.Optimizer,
):

    dataset = _resolve_dataset(data, gamma=parser_args.gamma)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    wlogger = WandbLogger(project="rgnet", name=data.exp_id, group="rl")

    thunder_module = LightningAdapter(
        gnn=gnn, actor_critic=agent, loss=loss, optim=optim
    )
    checkpoint_path = data.output_dir
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path
    )
    trainer = lightning.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=epochs,
        logger=wlogger,
        callbacks=[checkpoint_callback],
    )
    logging.info("Starting training")
    trainer.fit(thunder_module, loader)
    logging.info("Done")


def run():
    logging.getLogger().setLevel(logging.INFO)
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    time_stamp: str = datetime.now().strftime("%d-%m_%H-%M-%S")
    parser = create_args()
    parser_args = parser.parse_args()
    data = data_resolver.from_parser_args(parser_args, exp_id=time_stamp)

    requested_device = parser_args.device
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = torch.device(requested_device)

    gamma = parser_args.gamma

    embedding: EmbeddingModule = embedding_module.from_parser_args(
        parser_args, device=device, data_resolver=data
    )
    gnn: HeteroGNN = embedding.gnn

    agent: Agent = agent_module.from_parser_args(
        parser_args, data_resolver=data, embedding=embedding
    )

    actor_critic: ActorCritic = agent.actor.module
    assert isinstance(actor_critic, ActorCritic)

    value_estimator.from_parser_args(
        parser_args,
        data_resolver=data,
        device=device,
        loss=agent.loss,
        gamma=gamma,
        env=None,
        embedding=embedding,
    )
    optimizer = trainer_module._resolve_optim(parser_args, agent)
    epochs = getattr(parser_args, TrainerParameter.epochs)
    batch_size = parser_args.batch_size

    train(
        parser_args,
        data,
        actor_critic,
        gnn,
        batch_size=batch_size,
        device=requested_device,
        epochs=epochs,
        loss=agent.loss,
        optim=optimizer,
    )

    logging.info(f"Finished training. Saved under {data.output_dir}")


if __name__ == "__main__":
    # https://discuss.pytorch.org/t/training-fails-due-to-memory-exhaustion-when-running-in-a-python-multiprocessing-process/202773/2
    torch.multiprocessing.set_start_method("fork", force=True)
    run()
