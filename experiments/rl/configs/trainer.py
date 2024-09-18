import warnings
from argparse import ArgumentParser
from enum import StrEnum, auto
from math import ceil
from typing import Dict, List

import pymimir as mi
import torch
from torch.optim import SGD, Adam
from torchrl.envs import EnvBase
from torchrl.record.loggers import Logger

from experiments.rl.configs.agent import Agent
from experiments.rl.configs.value_estimator import ARGS_BOOL_TYPE, discounted_value
from experiments.rl.data_resolver import DataResolver
from rgnet.rl import RolloutCollector
from rgnet.rl.trainer import PolicyQuality, SupervisedValueLoss, Trainer


class Parameter(StrEnum):
    epochs = auto()
    batches_per_epoch = auto()
    rollout_length = auto()
    validation_interval = auto()
    validate_after_epoch = auto()
    log_interval = auto()
    save_trainer_interval = auto()
    clip_grad_norm = auto()

    optimizer = auto()
    learning_rate = auto()
    weight_decay = auto()
    lr_actor = auto()
    lr_critic = auto()
    lr_embedding = auto()


def optimal_values(space: mi.StateSpace, gamma: float):
    return torch.tensor(
        [
            discounted_value(space.get_distance_to_goal_state(s), gamma=gamma)
            for s in space.get_states()
        ],
        dtype=torch.float,
    )


def add_parser_args(parent_parser: ArgumentParser):
    parser = parent_parser.add_argument_group("Trainer")
    parser.add_argument(
        f"--{Parameter.epochs}",
        type=int,
        required=False,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        f"--{Parameter.batches_per_epoch}",
        type=int,
        required=False,
        default=None,
        help="Number of batches in one epoch."
        "Default will use the total number of states and batch size.",
    )
    parser.add_argument(
        f"--{Parameter.rollout_length}",
        type=int,
        required=False,
        default=1,
        help="Length of each trajectory for each batch, number of consecutive actions of the actor (default: 1)",
    )
    parser.add_argument(
        f"--{Parameter.validation_interval}",
        type=int,
        required=False,
        default=None,
        help="Interval in batches for validation (default: None)",
    )
    parser.add_argument(
        f"--{Parameter.validate_after_epoch}",
        type=ARGS_BOOL_TYPE,
        required=False,
        default=False,
        help="Whether to validate after each epoch (default: False)",
    )
    parser.add_argument(
        f"--{Parameter.log_interval}",
        type=int,
        required=False,
        default=10000,
        help="Interval for logging in frames (default: 10000)",
    )
    parser.add_argument(
        f"--{Parameter.save_trainer_interval}",
        type=int,
        required=False,
        default=10000,
        help="Interval for saving the trainer in frames (default: 10000)",
    )
    parser.add_argument(
        f"--{Parameter.clip_grad_norm}",
        type=bool,
        required=False,
        default=True,
        help="Whether to clip the gradient norm (default: True)",
    )

    # OPTIMIZER PARAMS
    parser.add_argument(
        f"--{Parameter.optimizer}",
        choices=["adam", "sgd"],
        required=False,
        default="adam",
        help="Optimizer to use (default: adam)",
    )
    parser.add_argument(
        f"--{Parameter.learning_rate}",
        type=float,
        required=False,
        default=2e-3,
        help="Learning rate (default: 2e-3)",
    )
    parser.add_argument(
        f"--{Parameter.weight_decay}",
        type=float,
        required=False,
        default=None,
        help="Weight decay (default will use the optimizer default)",
    )
    parser.add_argument(
        f"--{Parameter.lr_actor}",
        type=float,
        required=False,
        default=None,
        help="Specific learning rate for the actor(default: use learning_rate)",
    )
    parser.add_argument(
        f"--{Parameter.lr_critic}",
        type=float,
        required=False,
        default=None,
        help="Specific learning rate for the critic(default: use learning_rate)",
    )
    parser.add_argument(
        f"--{Parameter.lr_embedding}",
        type=float,
        required=False,
        default=None,
        help="Specific learning rate for the embedding(default: use learning_rate)",
    )
    return parent_parser


def _resolve_optim(parser_args, agent: Agent):
    optimizer_method = getattr(parser_args, Parameter.optimizer)
    learning_rate = getattr(parser_args, Parameter.learning_rate)
    weight_decay = getattr(parser_args, Parameter.weight_decay)
    lr_actor = getattr(parser_args, Parameter.lr_actor)
    lr_critic = getattr(parser_args, Parameter.lr_critic)
    lr_embedding = getattr(parser_args, Parameter.lr_embedding)

    if lr_actor is not None and agent.parameter_actor is None:
        warnings.warn(
            f"Learning rate for actor is set to {lr_actor} but the actor does not have learnable parameter."
        )
    if lr_critic is not None and agent.parameter_critic is None:
        warnings.warn(
            f"Learning rate for critic is set to {lr_critic} but the critic does not have learnable parameter."
        )
    if lr_embedding is not None and agent.parameter_embedding is None:
        warnings.warn(
            f"Learning rate for embedding is set to {lr_embedding} but the embedding does not have learnable parameter."
        )

    lr_parameters: List[Dict] = [
        {"params": agent.parameter_actor or [], "lr": lr_actor or learning_rate},
        {"params": agent.parameter_critic or [], "lr": lr_critic or learning_rate},
        {
            "params": agent.parameter_embedding or [],
            "lr": lr_embedding or learning_rate,
        },
    ]
    optim_parameter: Dict = {"params": lr_parameters}
    if weight_decay is not None:
        optim_parameter["weight_decay"] = weight_decay
    optimizer = Adam if optimizer_method == "adam" else SGD
    return optimizer(**optim_parameter)


def _resolve_collector(
    parser_args, data_resolver: DataResolver, agent: Agent, env: EnvBase
):
    batches_per_epoch = getattr(parser_args, Parameter.batches_per_epoch)
    if batches_per_epoch is None:
        total_states = sum([space.num_states() for space in data_resolver.spaces])
        batches_per_epoch = ceil(total_states / float(env.batch_size[0]))
    rollout_length = getattr(parser_args, Parameter.rollout_length)
    return RolloutCollector(
        environment=env,
        policy=agent.actor,
        rollout_length=rollout_length,
        num_batches=batches_per_epoch,
    )


def from_parser_args(
    parser_args, data_resolver: DataResolver, logger: Logger, agent: Agent, env: EnvBase
):
    optimizer = _resolve_optim(parser_args, agent)

    collector = _resolve_collector(
        parser_args, data_resolver=data_resolver, agent=agent, env=env
    )

    epochs = getattr(parser_args, Parameter.epochs)

    gamma = parser_args.gamma

    optimal_values_dict = {
        space: optimal_values(space, gamma) for space in data_resolver.spaces
    }

    class ValueModule(torch.nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.embedding = agent.embedding
            self.critic = agent.critic

        def forward(self, states: List[mi.State]):
            embedding = self.embedding(states)
            return self.critic(embedding)

    optimal_value_hook = SupervisedValueLoss(
        optimal_values=optimal_values_dict,
        value_module=ValueModule(),
        device=agent.embedding.device,
    )
    policy_precision_hook = PolicyQuality(data_resolver.spaces, agent.actor)

    return Trainer(
        collector=collector,
        epochs=epochs,
        optimizer=optimizer,
        logger=logger,
        clip_grad_norm=getattr(parser_args, Parameter.clip_grad_norm),
        optim_steps_per_batch=1,
        loss_module=agent.loss,
        save_trainer_interval=getattr(parser_args, Parameter.save_trainer_interval),
        log_interval=getattr(parser_args, Parameter.log_interval),
        eval_interval=getattr(parser_args, Parameter.validation_interval),
        validate_after_epoch=getattr(parser_args, Parameter.validate_after_epoch),
        eval_hooks=[optimal_value_hook, policy_precision_hook],
    )
