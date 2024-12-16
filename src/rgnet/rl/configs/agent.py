import dataclasses
from argparse import ArgumentParser
from enum import StrEnum, auto
from itertools import chain
from typing import Dict, Iterable, Optional

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch_geometric.nn import MLP
from torchrl.modules import ValueOperator

from rgnet.rl import (
    ActorCritic,
    ActorCriticLoss,
    EGreedyActorCriticHook,
    EGreedyModule,
    EmbeddingModule,
    EpsilonAnnealing,
)
from rgnet.rl.agents import OptimalPolicy, ValueModule
from rgnet.rl.configs.value_estimator import ARGS_BOOL_TYPE
from rgnet.rl.configs.value_estimator import Parameter as ValueEstimatorParameter
from rgnet.rl.data_layout import InputData
from rgnet.rl.envs.planning_env import PlanningEnvironment
from rgnet.rl.losses import CriticLoss
from rgnet.rl.losses.all_actions_ac_loss import AllActionsLoss


@dataclasses.dataclass
class Agent:
    critic: ValueOperator
    actor: Optional[TensorDictModule]
    embedding: EmbeddingModule
    loss: CriticLoss

    # We list the parameter explicitly as simply calling .parameters() might include some
    # parameters twice e.g. agent.policy has references to critic and embedding.
    # Additionally. this makes it clear that not all parts are trainable.
    parameter_critic: Optional[Iterable[torch.nn.Parameter]]
    parameter_actor: Optional[Iterable[torch.nn.Parameter]]
    parameter_embedding: Optional[Iterable[torch.nn.Parameter]]


class Parameter(StrEnum):
    algorithm = auto()
    value_net = auto()
    loss_critic_method = auto()
    batch_reduction_method = auto()
    use_epsilon_for_actor_critic = auto()


def simple_linear_net(hidden_size: int):
    return torch.nn.Linear(hidden_size, 1, bias=False)


def mlp_net(hidden_size: int):
    return MLP(
        channel_list=[hidden_size, hidden_size, 1],
        norm=None,
        dropout=0.0,
    )


def add_parser_args(parent_parser: ArgumentParser):
    parser = parent_parser.add_argument_group("Agent and Loss")
    parser.add_argument(
        f"--{Parameter.algorithm}",
        choices=["supervised", "egreedy", "actor_critic"],
        required=True,
        help="Type of agent to use (default: actor_critic)",
    )
    parser.add_argument(
        f"--{Parameter.value_net}",
        choices=["linear", "mlp"],
        required=False,
        default="mlp",
        help="Complexity of the value net (default: mlp)",
    )
    EpsilonAnnealing.add_parser_args(parent_parser)

    # Loss parameter
    parser.add_argument(
        f"--{Parameter.loss_critic_method}",
        choices=["l2", "l1"],
        required=False,
        default="l2",
        help="Loss function for the critic (default: l2)",
    )
    parser.add_argument(
        f"--{Parameter.batch_reduction_method}",
        choices=["mean", "sum"],
        required=False,
        default="mean",
        help="Reduction over the batch dimension (default: mean)",
    )
    parser.add_argument(
        f"--{Parameter.use_epsilon_for_actor_critic}",
        type=ARGS_BOOL_TYPE,
        required=False,
        default=False,
        help="Whether to use an epsilon for the actor critic (default: False)",
    )
    return parent_parser


def from_parser_args(
    parser_args,
    data_resolver: InputData,
    embedding: EmbeddingModule,
    env_keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
    agent_keys: ActorCritic.AcceptedKeys = ActorCritic.default_keys,
) -> Agent:
    config = AgentAndLossConfig(
        parser_args, data_resolver, embedding, env_keys, agent_keys
    )

    algorithm_choice = config.parser_values[Parameter.algorithm]

    if algorithm_choice == "supervised":
        agent = config._resolve_supervised()
    elif algorithm_choice == "egreedy":
        agent = config._resolve_egreedy()
    elif algorithm_choice == "actor_critic":
        agent = config._resolve_actor_critic()
    else:
        raise ValueError(f"Unknown algorithm choice: {algorithm_choice}")

    agent.critic.to(embedding.device)
    agent.actor.to(embedding.device)
    agent.loss.to(embedding.device)

    return agent


class AgentAndLossConfig:

    def __init__(
        self,
        parser_args,
        data_resolver: InputData,
        embedding: EmbeddingModule,
        env_keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
        agent_keys: ActorCritic.AcceptedKeys = ActorCritic.default_keys,
    ):
        self.data_resolver = data_resolver
        self.parser_args = parser_args
        self.parser_values: Dict = {p: getattr(parser_args, p) for p in Parameter}
        self.use_all_actions = getattr(
            parser_args, ValueEstimatorParameter.use_all_actions
        )
        self.epsilon_annealing = EpsilonAnnealing.from_parser_args(parser_args)
        self.value_net = (
            mlp_net(embedding.hidden_size)
            if self.parser_values[Parameter.value_net] == "mlp"
            else simple_linear_net(embedding.hidden_size)
        )
        self.embedding = embedding
        self.env_keys = env_keys
        self.agent_keys = agent_keys

    def _resolve_critic_loss(self, critic) -> CriticLoss:
        return CriticLoss(
            critic_network=critic,
            loss_critic_type=self.parser_values[Parameter.loss_critic_method],
            reduction=self.parser_values[Parameter.batch_reduction_method],
        )

    def _resolve_epsilon_module(self, policy, hook=None):
        epsilon_annealing = EpsilonAnnealing.from_parser_args(self.parser_args)
        # noinspection PyTypeChecker
        return TensorDictSequential(
            policy,
            EGreedyModule(
                epsilon_annealing,
                self.env_keys.transitions,
                self.env_keys.action,
                log_epsilon_actions=True,
                replace_action_hook=hook,
            ),
        )

    def _resolve_supervised(self) -> Agent:
        critic = ValueOperator(
            self.value_net,
            in_keys=[self.agent_keys.current_embedding],
            out_keys=[self.agent_keys.state_value],
        )
        actor = OptimalPolicy(self.data_resolver.spaces).as_td_module(
            state_key=self.env_keys.state, action_key=self.env_keys.action
        )
        loss = self._resolve_critic_loss(critic)
        return Agent(
            critic=critic,
            actor=actor,
            embedding=self.embedding,
            loss=loss,
            parameter_critic=critic.parameters(),
            parameter_actor=None,
            parameter_embedding=self.embedding.parameters(),
        )

    def _resolve_egreedy(self):

        agent = ValueModule(
            embedding=self.embedding,
            value_net=self.value_net,
        )
        # noinspection PyTypeChecker
        actor = self._resolve_epsilon_module(
            policy=agent.as_td_module(self.env_keys.transitions, self.env_keys.action),
        )

        critic = ValueOperator(
            self.value_net,
            in_keys=[self.agent_keys.current_embedding],
            out_keys=[self.agent_keys.state_value],
        )

        return Agent(
            critic=critic,
            actor=actor,
            embedding=self.embedding,
            loss=self._resolve_critic_loss(critic),
            parameter_critic=critic.parameters(),
            parameter_actor=None,
            parameter_embedding=self.embedding.parameters(),
        )

    def _resolve_actor_critic(self):
        agent = ActorCritic(
            hidden_size=self.embedding.hidden_size,
            embedding_module=self.embedding,
            value_net=self.value_net,
        )
        policy = agent.as_td_module(
            self.env_keys.state,
            self.env_keys.transitions,
            self.env_keys.action,
            add_probs=True,
        )
        if self.parser_values[Parameter.use_epsilon_for_actor_critic]:
            policy = self._resolve_epsilon_module(
                policy,
                hook=EGreedyActorCriticHook(agent.keys.probs, agent.keys.log_probs),
            )

        # includes policy, value_net and embeddings
        # use different learning rates for the policy and the value net

        loss_class = ActorCriticLoss
        if self.use_all_actions:
            loss_class = AllActionsLoss
        loss = loss_class(
            critic_network=agent.value_operator,
            reduction=self.parser_values[Parameter.batch_reduction_method],
            loss_critic_type=self.parser_values[Parameter.loss_critic_method],
        )
        return Agent(
            critic=agent.value_operator,
            actor=policy,
            embedding=self.embedding,
            loss=loss,
            parameter_critic=agent.value_operator.parameters(),
            parameter_actor=chain(
                agent.actor_net_probs.parameters(), agent.actor_net_probs.parameters()
            ),
            parameter_embedding=self.embedding.parameters(),
        )
