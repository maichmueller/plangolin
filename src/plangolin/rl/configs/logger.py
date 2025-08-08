from argparse import ArgumentParser
from enum import auto

try:
    from enum import StrEnum  # Available in Python 3.11+
except ImportError:
    from strenum import StrEnum  # Backport for Python < 3.11

import wandb
from torchrl.record.loggers import Logger

from plangolin.rl.configs.agent import Agent
from plangolin.rl.configs.value_estimator import ARGS_BOOL_TYPE
from plangolin.rl.data_layout import OutputData


class Parameter(StrEnum):
    logger_backend = auto()
    offline = auto()
    wandb_watch = auto()


def add_parser_args(parent_parser: ArgumentParser):
    parser = parent_parser.add_argument_group("Logger")
    parser.add_argument(
        f"--{Parameter.logger_backend}",
        choices=["csv", "tensorboard", "wandb"],
        required=False,
        default="wandb",
        help="Logger backend (default: wandb)",
    )
    parser.add_argument(
        f"--{Parameter.offline}",
        type=ARGS_BOOL_TYPE,
        required=False,
        default=False,
        help="Whether to sync to wandb services (default: False)",
    )
    parser.add_argument(
        f"--{Parameter.wandb_watch}",
        type=ARGS_BOOL_TYPE,
        required=False,
        default=False,
        help="Whether to watch the model (default: False)",
    )

    return parent_parser


def from_parser_args(parser_args, data_resolver: OutputData, agent: Agent) -> Logger:
    backend = getattr(parser_args, Parameter.logger_backend)

    exp_name = data_resolver.experiment_name

    if backend == "csv":
        from torchrl.record.loggers import CSVLogger

        logger = CSVLogger(exp_name=exp_name, log_dir=data_resolver.out_dir)

    elif backend == "tensorboard":
        from torchrl.record.loggers import TensorboardLogger

        logger = TensorboardLogger(exp_name=exp_name, log_dir=data_resolver.out_dir)

    elif backend == "wandb":
        from torchrl.record.loggers import WandbLogger

        logger = WandbLogger(
            exp_name=exp_name,
            offline=getattr(parser_args, Parameter.offline),
            save_dir=data_resolver.out_dir,
            project="plangolin",
            group="rl",
        )
        if getattr(parser_args, Parameter.wandb_watch):
            wandb.watch(agent, log="all")

    else:
        raise ValueError(f"Logger backend {backend} not supported.")

    logger.log_hparams(vars(parser_args))

    return logger
