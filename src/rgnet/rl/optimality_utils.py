from typing import Dict, List, Set

import torch
from tensordict import NestedKey
from torch import Tensor
from torchrl.modules import ValueOperator

import xmimir as xmi
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, tolist
from xmimir import XStateSpace


def optimal_policy(space: XStateSpace) -> Dict[int, Set[int]]:
    # index of state to set of indices of optimal actions
    # optimal[i] = {j},  0 <= j < len(space.iter_forward_transitions(space.get_states()[i]))
    optimal: Dict[int, Set[int]] = dict()
    for i, state in enumerate(space):
        transitions = list(space.forward_transitions(state))
        if not transitions:
            optimal[i] = set()
            continue
        best_distance = min(space.goal_distance(t.target) for t in transitions)
        best_actions: Set[int] = set(
            idx
            for idx, t in enumerate(transitions)
            if space.goal_distance(t.target) == best_distance
        )

        optimal[i] = best_actions
    return optimal


def optimal_policy_tensors(space: XStateSpace) -> list[Tensor]:
    # index of state to distribution over successor states as ordered in the state space object
    optimal: list[Tensor] = [None] * len(space)
    for i, state in enumerate(space):
        transitions = list(space.forward_transitions(state))
        distances = torch.tensor(
            [space.goal_distance(t.target) for t in transitions],
            dtype=torch.int,
        )
        best_distances, _ = torch.where(distances == torch.min(distances))
        policy = torch.zeros((len(transitions),), dtype=torch.float)
        policy[best_distances] = 1.0 / len(best_distances)
        optimal[i] = policy
    return optimal


def discounted_value(distance_to_goal, gamma):
    return -(1 - gamma**distance_to_goal) / (1 - gamma)


def optimal_discounted_values(space: XStateSpace, gamma: float):
    return torch.tensor(
        [discounted_value(space.goal_distance(s), gamma=gamma) for s in space],
        dtype=torch.float,
    )


class OptimalValueFunction(torch.nn.Module):
    """Don't predict the value target just use the discounted distance to goal"""

    def __init__(self, optimal_values: Dict[xmi.XState, float], device: torch.device):
        super().__init__()
        self.optimal_values = optimal_values
        self.device = device

    def __call__(
        self, batched_states: List[xmi.XState] | NonTensorWrapper
    ) -> torch.Tensor:
        batched_states = tolist(batched_states)
        return torch.stack(
            [
                torch.tensor(
                    [self.optimal_values[state] for state in states],
                    dtype=torch.float,
                    device=self.device,
                ).view(-1, 1)
                for states in batched_states
            ]
        )

    def as_td_module(
        self, state_key: NestedKey, state_value_key: NestedKey
    ) -> ValueOperator:
        return ValueOperator(
            module=self, in_keys=[state_key], out_keys=[state_value_key]
        )
