from argparse import ArgumentParser
from distutils.util import strtobool
from enum import StrEnum, auto
from typing import Optional

import torch
from torchrl.objectives import ValueEstimators

from rgnet.rl import ActorCritic, EmbeddingModule, NonTensorTransformedEnv
from rgnet.rl.data_layout import InputData
from rgnet.rl.envs.planning_env import PlanningEnvironment
from rgnet.rl.losses import CriticLoss
from rgnet.rl.losses.optimal_value_function import OptimalValueFunction
from rgnet.rl.optimality_utils import discounted_value

ARGS_BOOL_TYPE = lambda x: bool(strtobool(str(x)))


class Parameter(StrEnum):
    optimal_values = auto()
    use_all_actions = auto()


def from_parser_args(
    parser_args,
    data_resolver: InputData,
    device: torch.device,
    loss: CriticLoss,
    gamma: float,
    env: Optional[NonTensorTransformedEnv] = None,  # only needed for all_actions
    embedding: Optional[EmbeddingModule] = None,  # only needed for all_actions
    env_keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
):
    use_optimal_values = getattr(parser_args, Parameter.optimal_values)
    use_all_actions = getattr(parser_args, Parameter.use_all_actions)

    if use_optimal_values:
        state_values = {}
        for space in data_resolver.spaces:
            state_values.update(
                {s: space.get_distance_to_goal_state(s) for s in space.get_states()}
            )
        discounted_state_values = {
            s: discounted_value(v, gamma) for s, v in state_values.items()
        }
        ovf = OptimalValueFunction(
            optimal_values=discounted_state_values, device=device
        )
        value_operator = ovf.as_td_module(
            state_key=env_keys.state,
            state_value_key=ActorCritic.default_keys.state_value,
        )
        loss.make_value_estimator(
            value_type=ValueEstimators.TD0,
            optimal_targets=value_operator,
            gamma=parser_args.gamma,
            shifted=True,
        )
    elif use_all_actions:
        assert env is not None
        assert embedding is not None
        loss.make_value_estimator(
            value_type="AllActionsValueEstimator",
            env=env,
            embedding_module=embedding,
            gamma=parser_args.gamma,
        )
    else:
        loss.make_value_estimator(
            ValueEstimators.TD0,
            gamma=parser_args.gamma,
            shifted=True,
        )


def add_parser_args(
    parent_parser: ArgumentParser,
):
    parser = parent_parser.add_argument_group("Value Estimator")
    parser.add_argument(
        f"--{Parameter.optimal_values}",
        type=ARGS_BOOL_TYPE,
        required=False,
        default=False,
        help="Whether the value estimator should use optimal values for the value-targets (default: False)."
        "This is only possible if you can build the state spaces.",
    )
    parser.add_argument(
        f"--{Parameter.use_all_actions}",
        type=ARGS_BOOL_TYPE,
        required=False,
        default=False,
        help="Whether the value estimator should consider all actions to compute the expected value (default: False).",
    )
    return parent_parser
