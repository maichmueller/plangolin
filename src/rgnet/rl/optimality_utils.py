from typing import Dict, Set

import pymimir as mi
import torch
from torch import Tensor


def optimal_policy(space: mi.StateSpace) -> Dict[int, Set[int]]:
    # index of state to set of indices of optimal actions
    # optimal[i] = {j},  0 <= j < len(space.get_forward_transitions(space.get_states()[i]))
    optimal: Dict[int, Set[int]] = dict()
    for i, state in enumerate(space.get_states()):
        if len(space.get_forward_transitions(state)) == 0:
            optimal[i] = set()
            continue
        best_distance = min(
            space.get_distance_to_goal_state(t.target)
            for t in space.get_forward_transitions(state)
        )
        best_actions: Set[int] = set(
            idx
            for idx, t in enumerate(space.get_forward_transitions(state))
            if space.get_distance_to_goal_state(t.target) == best_distance
        )

        optimal[i] = best_actions
    return optimal


def optimal_policy_tensors(space: mi.StateSpace) -> list[Tensor]:
    # index of state to distribution over successor states as ordered in the state space object
    optimal: list[Tensor] = [None] * space.num_states()
    for i, transitions in enumerate(
        map(space.get_forward_transitions, space.get_states())
    ):
        distances = torch.tensor(
            [space.get_distance_to_goal_state(t.target) for t in transitions],
            dtype=torch.int,
        )
        best_distances, _ = torch.where(distances == torch.min(distances))
        policy = torch.zeros((len(transitions),), dtype=torch.float)
        policy[best_distances] = 1.0 / len(best_distances)
        optimal[i] = policy
    return optimal


def discounted_value(distance_to_goal, gamma):
    return -(1 - gamma**distance_to_goal) / (1 - gamma)


def optimal_discounted_values(space: mi.StateSpace, gamma: float):
    return torch.tensor(
        [
            discounted_value(space.get_distance_to_goal_state(s), gamma=gamma)
            for s in space.get_states()
        ],
        dtype=torch.float,
    )
