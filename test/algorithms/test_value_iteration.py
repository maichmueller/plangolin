from test.fixtures import fresh_flashdrive, medium_blocks  # noqa: F401, F403

import pytest
import torch

from plangolin.algorithms import (
    ValueIterationMP,
    bellman_optimal_values,
    mdp_graph_as_pyg_data,
    optimal_policy,
)
from plangolin.rl.envs import ExpandedStateSpaceEnv
from plangolin.rl.reward import UnitReward


@pytest.mark.parametrize(
    "env_representation",
    [
        "pyg",
        "networkx",
    ],
)
def test_mp_on_optimal_medium(env_representation, medium_blocks):
    space, _, _ = medium_blocks
    gamma = 0.9
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=gamma),
        reset=True,
    )

    value_iteration_mp = ValueIterationMP(
        gamma, num_iterations=100, difference_threshold=0.001
    )

    optimal_policy_dict = optimal_policy(space)

    def optimal_probs(i, state):
        optimal_action_idx = next(iter(optimal_policy_dict[i]))
        probs = torch.zeros(
            (len(list(space.forward_transitions(state))),),
            dtype=torch.float,
        )
        probs[optimal_action_idx] = 1.0
        return probs

    optimal_policy_probabilities: tuple[torch.Tensor, ...] = tuple(
        optimal_probs(i, s) for (i, s) in enumerate(space)
    )
    if env_representation == "networkx":
        graph = env.to_mdp_graph(0, optimal_policy_probabilities)
        graph = mdp_graph_as_pyg_data(graph)
    else:
        graph = env.to_pyg_data(0, optimal_policy_probabilities)

    values = value_iteration_mp(graph)

    # this uses the value iteration MP only with the env. For a state space object, it will use the
    # contained distance to the goal for each state.
    optimal_values = bellman_optimal_values(space, gamma=gamma)
    values = values.squeeze()
    optimal_values = optimal_values.squeeze()
    assert values.shape == optimal_values.shape
    assert torch.allclose(values, optimal_values, 0.01)
