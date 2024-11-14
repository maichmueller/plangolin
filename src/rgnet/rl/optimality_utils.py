from typing import Dict, Set

import pymimir as mi
import torch


def optimal_policy(space: mi.StateSpace) -> Dict[int, Set[int]]:
    # index of state to set of indices of optimal actions
    # optimal[i] = {j},  0 <= j < len(space.get_forward_transitions(space.get_states()[i]))
    optimal: Dict[int, Set[int]] = dict()
    for i, state in enumerate(space.get_states()):
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
