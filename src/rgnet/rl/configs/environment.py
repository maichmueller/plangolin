from argparse import ArgumentParser
from enum import auto

from rgnet.rl.reward import UnitReward

try:
    from enum import StrEnum  # Available in Python 3.11+
except ImportError:
    from strenum import StrEnum  # Backport for Python < 3.11

import torch

from rgnet.rl.data_layout import InputData
from rgnet.rl.envs import MultiInstanceStateSpaceEnv
from rgnet.rl.envs.expanded_state_space_env import IteratingReset, WeightedRandomReset


class Parameter(StrEnum):
    batch_size = auto()
    seed = auto()


def from_parser_args(
    parser_args, data_resolver: InputData, device: torch.device, gamma: float
) -> MultiInstanceStateSpaceEnv:
    spaces = data_resolver.spaces
    dead_end_reward = -1.0 / (1.0 - gamma)
    batch_size = getattr(parser_args, Parameter.batch_size)
    seed = getattr(parser_args, Parameter.batch_size)
    # test if all spaces in the list are the same object
    env = MultiInstanceStateSpaceEnv(
        spaces,
        reset_strategy=IteratingReset(),
        batch_size=torch.Size((batch_size,)),
        seed=seed,
        device=device,
        reward_function=UnitReward(deadend_reward=dead_end_reward),
    )
    env.make_replacement_strategy(WeightedRandomReset)

    return env


def add_parser_args(parent_parser: ArgumentParser):
    parser = parent_parser.add_argument_group("Environment Setup")
    parser.add_argument(
        f"--{Parameter.batch_size.value}",
        type=int,
        required=False,
        default=32,
        help="Batch size for the environment (default: 32)",
    )
    parser.add_argument(
        f"--{Parameter.seed.value}",
        type=int,
        required=False,
        default=42,
        help="Seed for the environment (default: 42)",
    )

    return parent_parser
