from typing import Dict, Mapping, Set

import torch
from torch import Tensor

from rgnet.rl.envs import ExpandedStateSpaceEnv
from xmimir import XState, XStateSpace


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
