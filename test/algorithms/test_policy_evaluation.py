import warnings
from test.fixtures import fresh_flashdrive, medium_blocks  # noqa: F401, F403

import pytest
import torch

from rgnet.algorithms import PolicyEvaluationMP, bellman_optimal_values, optimal_policy
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.reward import UnitReward
from rgnet.utils import mdp_graph_as_pyg_data


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
    )
    env.reset()

    policy_eval_mp = PolicyEvaluationMP(
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
    graph = env.to_mdp_graph(0, optimal_policy_probabilities)
    graph = mdp_graph_as_pyg_data(graph)

    values = policy_eval_mp(graph)

    optimal_values = bellman_optimal_values(space, gamma=gamma)
    values = values.squeeze()
    optimal_values = optimal_values.squeeze()
    assert torch.allclose(values, optimal_values, 0.01)


@pytest.mark.parametrize(
    "env_representation",
    [
        "pyg",
        "networkx",
    ],
)
def test_mp_on_faulty_medium(env_representation, medium_blocks):
    """
    Test that running policy evaluation on a policy that never reaches the goal will yield
    discounted infinite trajectory values for all non-goal states.
    """
    space, _, _ = medium_blocks

    gamma = 0.9
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=gamma),
    )
    env.reset()
    policy_eval_mp = PolicyEvaluationMP(
        gamma, num_iterations=100, difference_threshold=0.001
    )
    goal_state = next(space.goal_states_iter())
    one_before_goal = list(space.backward_transitions(goal_state))
    assert len(one_before_goal) == 1, (
        "Test assumption violated."
        " Medium blocks problem should only have one goal state with one predecessor."
    )
    one_before_goal = one_before_goal[0].source
    one_before_goal_idx = one_before_goal.index

    # make sure the goal is never reached
    def faulty_probs(i, s):
        nr_transitions = space.forward_transition_count(s)
        probs = torch.rand(
            (nr_transitions,),
            dtype=torch.float,
            generator=torch.Generator().manual_seed(123456789),
        )
        if i == one_before_goal_idx:
            probs = torch.zeros((nr_transitions,), dtype=torch.float)
            for i, succ in enumerate(space.forward_transitions(s)):
                if succ.target != goal_state:
                    probs[i] = 1.0 / nr_transitions
        return probs.abs() / probs.abs().sum()

    probs_list = tuple(faulty_probs(i, s) for (i, s) in enumerate(space))

    graph = env.to_mdp_graph(0, probs_list)
    graph_data = mdp_graph_as_pyg_data(graph)

    _debug_policy_per_state = {
        node: sum(
            graph_data.edge_attr[graph_data.edge_index[1, :] == node.index, 0].tolist()
        )
        for node in graph.nodes
    }
    if not all(abs(s - 1.0) < 1e-5 for s in _debug_policy_per_state.values()):
        warnings.warn("Policy does not sum up to 1 for all states.")

    # _debug_plotit(graph_data, goal_state)

    values = policy_eval_mp(graph_data)

    # The goal is never reached; therefore, the values for all states should go towards
    # the discounted infinite trajectory length, which is -1 / (1-gamma).
    expected_values = torch.full((len(space),), -1 / (1 - gamma))
    expected_values[goal_state.index] = 0.0
    assert torch.allclose(values, expected_values, atol=0.01)
    # We can never go beyond -1 / (1 gamma).
    assert (values >= -1.0 / (1.0 - gamma)).all()
