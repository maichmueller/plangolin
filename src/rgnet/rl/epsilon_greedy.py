import abc
import dataclasses
import enum
from argparse import ArgumentParser
from typing import Any, List, Optional, Tuple

import pymimir as mi
import torch.nn
from tensordict import NestedKey, NonTensorStack, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.nn import MLP
from torchrl.envs.utils import ExplorationType, exploration_type

from rgnet.rl import EmbeddingModule
from rgnet.rl.agent import Agent, embed_transition_targets
from rgnet.rl.envs.planning_env import PlanningEnvironment
from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


class EpsilonAnnealing:
    class Parameter(enum.StrEnum):
        EPSILON_INIT = "epsilon_init"
        EPSILON_END = "epsilon_end"
        ANNEALING_STEPS = "annealing_steps"

    def __init__(self, epsilon_init: float, epsilon_end: float, annealing_steps: float):
        assert 0 <= epsilon_init <= 1, "epsilon_init must be in [0, 1]"
        assert 0 <= epsilon_end <= 1, "epsilon_end must be in [0, 1]"
        assert (
            epsilon_init >= epsilon_end
        ), "epsilon_init must be greater than or equal to epsilon_end"
        self.epsilon = epsilon_init
        self.eps_init = epsilon_init
        self.eps_end = epsilon_end
        self.annealing_steps = annealing_steps
        self.eps_step = (epsilon_init - epsilon_end) / annealing_steps

    def step_epsilon(self):
        self.epsilon = max(
            self.eps_end,
            (self.epsilon - self.eps_step),
        )

    @staticmethod
    def from_parser_args(parser_args):
        kwargs = {
            p.value: getattr(parser_args, p.value) for p in EpsilonAnnealing.Parameter
        }
        return EpsilonAnnealing(**kwargs)

    @staticmethod
    def add_parser_args(
        parent_parser: ArgumentParser,
    ):
        parser = parent_parser.add_argument_group("Epsilon Annealing")
        parser.add_argument(
            f"--{EpsilonAnnealing.Parameter.EPSILON_INIT.value}",
            type=float,
            required=False,
            default=0.5,
            help="Initial epsilon value (default: 0.5)",
        )
        parser.add_argument(
            f"--{EpsilonAnnealing.Parameter.EPSILON_END.value}",
            type=float,
            required=False,
            default=0.01,
            help="Final epsilon value at the end of annealing (default: 0.01).",
        )
        parser.add_argument(
            f"--{EpsilonAnnealing.Parameter.ANNEALING_STEPS.value}",
            type=int,
            required=False,
            default=1000,
            help="The number of iterations over which the epsilon value is annealed "
            "(default: 1000).",
        )
        return parent_parser


class EGreedyAgent(torch.nn.Module):
    @dataclasses.dataclass(frozen=True)
    class AcceptedKeys:
        # If log_epsilon_actions is True, the key will be used to store the epsilon action.
        # epsilon_action[batch_idx] = True if the action was sampled from the epsilon greedy policy.
        epsilon_action_key = "epsilon_action"

    default_keys = AcceptedKeys()

    def __init__(
        self,
        epsilon_manager: EpsilonAnnealing,
        embedding: EmbeddingModule,
        log_epsilon_actions: bool = False,
        value_net: torch.nn.Module | None = None,
        keys: AcceptedKeys = default_keys,
    ):
        super().__init__()
        self.epsilon_manager = epsilon_manager
        self.log_epsilon_actions = log_epsilon_actions
        self.keys = keys
        self._embedding_module = embedding
        if value_net is None:
            value_net = MLP(
                channel_list=[
                    self._embedding_module.hidden_size,
                    self._embedding_module.hidden_size,
                    1,
                ],
                norm=None,
                dropout=0.0,
            )
        self.value_net = value_net

    def _internal_forward(
        self, transitions: List[List[mi.Transition]] | NonTensorWrapper
    ) -> List[mi.Transition]:

        with torch.no_grad():
            # We don't want gradient for the next values and next embeddings.
            # The value net will be updated by a ValueEstimator like TD0Estimator.
            successor_embeddings = embed_transition_targets(
                transitions, self._embedding_module
            )
            successor_values: List[torch.Tensor] = [
                self._value_net(e) for e in successor_embeddings
            ]
            indices_of_best: List[int] = [
                torch.argmax(sv, dim=0).item() for sv in successor_values
            ]
            actions: List[mi.Transition] = [
                ts[indices_of_best]
                for (idx_of_best, ts) in zip(indices_of_best, transitions)
            ]
        return actions

    def forward(
        self, transitions: List[mi.Transition] | NonTensorWrapper
    ) -> NonTensorStack:

        transitions = non_tensor_to_list(transitions)
        actions = self._internal_forward(transitions)

        if (
            exploration_type() != ExplorationType.RANDOM
            and exploration_type() is not None
        ):
            return as_non_tensor_stack(actions)

        batch_size = len(transitions)
        random_steps = torch.rand(size=(batch_size,)) < self.epsilon_manager.epsilon

        for idx, should_replace in enumerate(random_steps):
            if should_replace:
                sampled_action_idx = torch.randint(
                    0, len(transitions[idx]), (1,)
                ).item()
                actions[idx] = transitions[idx][sampled_action_idx]

        self.epsilon_manager.step_epsilon()

        if self.log_epsilon_actions:
            as_non_tensor_stack(actions), random_steps

        return as_non_tensor_stack(actions)

    def as_td_module(self, transitions_key, actions_key):
        out_keys = [actions_key]
        if self.log_epsilon_actions:
            out_keys.append(self.keys.epsilon_action_key)
        return TensorDictModule(
            module=self, in_keys=[transitions_key], out_keys=out_keys
        )


class EGreedyActorCritic(Agent):

    def __init__(
        self,
        epsilon_annealing: EpsilonAnnealing,
        embedding_module: EmbeddingModule,
        value_net: torch.nn.Module | None = None,
        keys: Agent.AcceptedKeys = Agent.default_keys,
    ):
        super().__init__(embedding_module, value_net, keys)
        self.epsilon_manager = epsilon_annealing

    def forward(
        self,
        state: NonTensorWrapper | List[mi.State],
        transitions: NonTensorWrapper | List[List[mi.Transition]],
        current_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[NonTensorStack, Tensor | Any, Tensor, NonTensorStack, Tensor]:

        transitions = non_tensor_to_list(transitions)
        if current_embedding is None:
            current_embedding = self._embedding_module(state)
        successor_embeddings = embed_transition_targets(
            transitions, self._embedding_module
        )
        # len(batched_probs) == batch_size, batched_probs[i].shape == len(transitions[i])
        batched_probs: list[Tensor] = self._actor_probs(
            current_embedding, successor_embeddings
        )

        action_indices, log_probs = self._sample_distribution(batched_probs)
        actions = self._select_action(action_indices, transitions)

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            random_steps = (
                torch.rand(size=(len(transitions),)) < self.epsilon_manager.epsilon
            )
            for idx, should_replace in enumerate(random_steps):
                if should_replace:
                    sampled_action_idx = torch.randint(0, len(transitions[idx]), (1,))
                    actions[idx] = transitions[idx][sampled_action_idx.item()]
                    new_log_probs = Categorical(probs=batched_probs[idx]).log_prob(
                        sampled_action_idx
                    )
                    assert new_log_probs.requires_grad
                    log_probs[idx] = new_log_probs

            self.epsilon_manager.step_epsilon()
        else:
            random_steps = torch.zeros(len(transitions), dtype=torch.bool)

        return (
            as_non_tensor_stack(actions),
            current_embedding,
            log_probs,
            as_non_tensor_stack(batched_probs),
            random_steps,
        )

    def as_td_module(
        self, state_key: NestedKey, transition_key: NestedKey, action_key: NestedKey
    ):
        out_keys = [
            action_key,
            self._keys.current_embedding,
            self._keys.log_probs,
            self._keys.probs,
            EGreedyAgent.default_keys.epsilon_action_key,
        ]
        return TensorDictModule(
            module=self,
            in_keys=[state_key, transition_key, self._keys.current_embedding],
            out_keys=out_keys,
        )
