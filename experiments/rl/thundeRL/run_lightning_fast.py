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
from rgnet.rl.thundeRL.collate import Collate
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
        / "dataset"
        / data_resolver.domain.name
    )
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
    torch.multiprocessing.set_sharing_strategy("file_system")

    dataset = _resolve_dataset(data, gamma=parser_args.gamma)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=Collate(),
        batch_size=batch_size,
        shuffle=True,
    )
    wlogger = WandbLogger(project="rgnet", name=data.exp_id, group="rl", offline=True)

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


def run():
    logging.getLogger().setLevel(logging.INFO)
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
    run()
