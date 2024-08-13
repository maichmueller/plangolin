import dataclasses
import enum
from argparse import ArgumentParser
from typing import Callable, List

import pymimir as mi
import torch.nn
from tensordict import NestedKey, TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torch.distributions import Categorical
from torch_geometric.nn import MLP
from torchrl.envs.utils import ExplorationType, exploration_type

from rgnet.rl import EmbeddingModule
from rgnet.rl.agent import embed_transition_targets
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


class EGreedyModule(TensorDictModuleBase):
    """We can't use the EGreedyModule from torchrl as they assume that the action-spec
    can generate arbitrary actions without the current state, which is not the case for
    planning problems in which the set of applicable actions depends on the current state.
    """

    @dataclasses.dataclass(frozen=True)
    class AcceptedKeys:
        # If log_epsilon_actions is True, the key will be used to store the epsilon action.
        # epsilon_action[batch_idx] = True if the action was sampled from the epsilon greedy policy.
        epsilon_action_key = "epsilon_action"

    default_keys = AcceptedKeys()

    def __init__(
        self,
        epsilon_annealing: EpsilonAnnealing,
        transitions_key: NestedKey,
        actions_key: NestedKey,
        log_epsilon_actions: bool = False,
        replace_action_hook: (
            Callable[[TensorDict, int, torch.Tensor], None] | None,
        ) = None,
        keys: AcceptedKeys = default_keys,
    ):
        super().__init__()
        self.epsilon_manager = epsilon_annealing
        self.log_epsilon_actions = log_epsilon_actions
        self.replace_action_hook = replace_action_hook
        self.keys = keys
        self.actions_key = actions_key
        self.transitions_key = transitions_key
        self.in_keys = [actions_key, transitions_key]
        if not self.log_epsilon_actions:
            self.out_keys = [actions_key, self.keys.epsilon_action_key]
        else:
            self.out_keys = [self.keys.epsilon_action_key]

    def forward(self, tensordict):

        if (
            exploration_type() != ExplorationType.RANDOM
            and exploration_type() is not None
        ):
            return tensordict

        transitions = tensordict[self.transitions_key]
        batch_size = len(transitions)
        random_steps = torch.rand(size=(batch_size,)) < self.epsilon_manager.epsilon

        for idx, should_replace in enumerate(random_steps):
            if should_replace:
                sampled_action_idx = torch.randint(0, len(transitions[idx]), (1,))
                new_sampled_action = transitions[idx][sampled_action_idx.item()]
                tensordict[self.actions_key][idx] = new_sampled_action
                if self.replace_action_hook is not None:
                    self.replace_action_hook(tensordict, idx, sampled_action_idx)

        self.epsilon_manager.step_epsilon()  # only one step per batch (debatable)

        if self.log_epsilon_actions:
            tensordict[self.keys.epsilon_action_key] = random_steps

        return tensordict


class ValueModule(torch.nn.Module):

    def __init__(
        self,
        embedding: EmbeddingModule,
        value_net: torch.nn.Module | None = None,
    ):
        super().__init__()
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

    def forward(
        self, transitions_in: List[List[mi.Transition]] | NonTensorWrapper
    ) -> List[mi.Transition] | NonTensorWrapper:
        transitions: List[List[mi.Transition]] = non_tensor_to_list(transitions_in)
        with torch.no_grad():
            # We don't want gradient for the next values and next embeddings.
            # The value net will be updated by a ValueEstimator like TD0Estimator.
            successor_embeddings = embed_transition_targets(
                transitions, self._embedding_module
            )
            successor_values: List[torch.Tensor] = [
                self.value_net(e) for e in successor_embeddings
            ]
            indices_of_best: List[int] = [
                torch.argmax(sv, dim=0).item() for sv in successor_values
            ]
            actions: List[mi.Transition] = [
                ts[idx_of_best]
                for (idx_of_best, ts) in zip(indices_of_best, transitions)
            ]
        return as_non_tensor_stack(actions)

    def as_td_module(self, transitions_key, actions_key):
        return TensorDictModule(
            module=self, in_keys=[transitions_key], out_keys=[actions_key]
        )


class EGreedyActorCritic:

    def __init__(self, probs_key: NestedKey, log_probs_key: NestedKey):
        self.probs_key = probs_key
        self.log_probs_key = log_probs_key

    def __call__(self, tensordict, action_index, sampled_new_action_index):
        batched_probs = tensordict[self.probs_key]
        new_log_probs = Categorical(probs=batched_probs[action_index]).log_prob(
            sampled_new_action_index
        )
        assert new_log_probs.requires_grad
        tensordict[self.log_probs_key][action_index] = new_log_probs
