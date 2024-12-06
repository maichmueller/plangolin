import itertools
from test.fixtures import fresh_drive, medium_blocks
from typing import List

import pymimir as mi
import torch

from rgnet.rl.optimality_utils import optimal_discounted_values, optimal_policy
from rgnet.rl.thundeRL.policy_evaluation import (
    PolicyEvaluationMessagePassing,
    build_mdp_graph_with_prob,
    mdp_graph_as_pyg_data,
)


def _placeholder_probs(space: mi.StateSpace):
    # Contains 0...(|Edges| - 1) as probabilities
    it = itertools.count()
    return [
        torch.Tensor(
            [next(it) for _ in range(len(space.get_forward_transitions(state)))],
        )
        for state in space.get_states()
    ]


def test_build_mdp_graph(medium_blocks):
    space: mi.StateSpace = medium_blocks[0]
    nx_graph = build_mdp_graph_with_prob(space, _placeholder_probs(space))
    assert all(
        all(
            out_edge[0] == space.get_unique_id(s)
            and out_edge[1] == space.get_unique_id(out_transition.target)
            for out_edge, out_transition in zip(
                nx_graph.out_edges(nbunch=[i]), space.get_forward_transitions(s)
            )
        )
        for i, s in enumerate(space.get_states())
    )


def test_mdp_graph_as_pyg_data(medium_blocks):
    space: mi.StateSpace = medium_blocks[0]
    probs_list = _placeholder_probs(space)
    pyg_graph = mdp_graph_as_pyg_data(build_mdp_graph_with_prob(space, probs_list))
    assert (pyg_graph.edge_attr[:, 0] == torch.cat(probs_list)).all()


def test_mp_on_optimal_medium(fresh_drive, medium_blocks):
    space, _, _ = medium_blocks

    gamma = 0.9

    value_iteration_mp = PolicyEvaluationMessagePassing(
        gamma, num_iterations=100, difference_threshold=0.001
    )

    optimal_policy_dict = optimal_policy(space)

    def optimal_probs(i, state):
        optimal_action_idx = next(iter(optimal_policy_dict[i]))
        probs = torch.zeros(
            (len(space.get_forward_transitions(state)),), dtype=torch.float
        )
        probs[optimal_action_idx] = 1.0
        return probs

    optimal_policy_probabilities: List[torch.Tensor] = [
        optimal_probs(i, s) for (i, s) in enumerate(space.get_states())
    ]
    graph = build_mdp_graph_with_prob(space, optimal_policy_probabilities)
    graph = mdp_graph_as_pyg_data(graph)

    values = value_iteration_mp(graph)

    optimal_values = optimal_discounted_values(space, gamma)

    assert torch.allclose(values, optimal_values, 0.01)


def test_mp_on_faulty_medium(fresh_drive, medium_blocks):
    """
    Test that running policy evaluation on a policy that never reaches the goal will yield
    discounted infinite trajectory values for all non-goal states.
    """
    space, _, _ = medium_blocks

    gamma = 0.9

    value_iteration_mp = PolicyEvaluationMessagePassing(
        gamma, num_iterations=100, difference_threshold=0.01
    )
    goal_state = space.get_goal_states()[0]
    one_before_goal = space.get_backward_transitions(goal_state)
    assert len(one_before_goal) == 1, (
        "Test assumption violated."
        " Medium blocks problem should only have one goal state with one predecessor."
    )
    one_before_goal = one_before_goal[0].source
    one_before_goal_idx = space.get_unique_id(one_before_goal)

    # make sure the goal is never reached
    def faulty_probs(i, s):
        transitions = space.get_forward_transitions(s)
        probs = torch.rand((len(transitions),), dtype=torch.float).softmax(dim=-1)
        if i == one_before_goal_idx:
            probs = torch.zeros((len(transitions),), dtype=torch.float)
            probs[1] = 1.0
        return probs

    probs_list = [faulty_probs(i, s) for (i, s) in enumerate(space.get_states())]

    graph = build_mdp_graph_with_prob(space, probs_list)
    graph = mdp_graph_as_pyg_data(graph)

    values = value_iteration_mp(graph)

    # The goal is never reached; therefore, the values for all states should go towards
    # the discounted infinite trajectory length, which is -1 / (1-gamma).
    expected_values = torch.full((space.num_states(),), -1 / (1 - gamma))
    expected_values[space.get_unique_id(goal_state)] = 0.0
    assert torch.allclose(values, expected_values, atol=0.01)
    # We can never go beyond -1 / (1 gamma).
    #
    assert (values >= -10).all()
