from typing import Dict, Mapping, Sequence, Set

import numpy as np
import torch
from torch import Tensor

from rgnet.rl.envs import ExpandedStateSpaceEnv, PlanningEnvironment
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


def value_iteration(
    space: XStateSpace, gamma: float, num_iterations: int, difference_threshold: float
):
    # Initialize values to 0
    values = torch.zeros((len(space),), dtype=torch.float)
    for _ in range(num_iterations):
        new_values = torch.zeros((len(space),), dtype=torch.float)
        for i, state in enumerate(space):
            transitions = list(space.forward_transitions(state))
            if not transitions:
                new_values[i] = 0.0
                continue
            new_values[i] = torch.max(
                torch.tensor(
                    [
                        torch.sum(
                            optimal_discounted_values(space, gamma)
                            * optimal_policy_tensors(space)[i]
                        )
                    ]
                )
            )


def policy_evaluation(
    policy: Mapping[XState, Tensor],
    env: ExpandedStateSpaceEnv,
    gamma: float,
    num_iterations: int,
    goal_value: float = 0.0,
):
    """
    Evaluates a given policy using iterative policy evaluation.

    The value function V is computed by repeatedly applying the Bellman expectation backup:

    .. math::
        V(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) ( R(s,a,s') + \gamma  V(s') )

    with deterministic actions, hence successor states as actions, this simplifies to

    .. math::
        V(s) = \sum_{s'} \pi(s'|s) ( R(s,s') + \gamma  V(s') )

    Args:
        policy: A mapping from each state to a list of action probabilities.
        env: The environment to evaluate the policy in.
        gamma: The discount factor (0 <= gamma <= 1).
        num_iterations: The number of iterations to perform for evaluation.
        goal_value: The value to assign to goal states.

    Returns:
        A dictionary mapping each state to its evaluated value.
    """
    space = env.reset()["instance"][0]
    nr_states = len(space)
    # Initialize the value function for all states to zero.
    V = torch.zeros((nr_states,), dtype=torch.float)

    for iteration in range(num_iterations):
        for state in space:
            if space.is_goal(state):
                V[state.index] = goal_value
                continue

            action_probabilities = policy[state]
            transitions = list(space.forward_transitions(state))
            reward, done = env.get_reward_and_done(transitions, current_states=(state,))
            transitions = torch.tensor(
                tuple(t.target.index for t in transitions), dtype=torch.int
            )
            # expected value for the state
            v = torch.dot(
                action_probabilities, (reward + gamma * V[transitions]) * (~done)
            )
            # we update values mid-iteration, so that other states may already use the updated values
            # this is not the same as the original algorithm, but it is more efficient, and the results are the same
            V[state.index] = v
    return V
