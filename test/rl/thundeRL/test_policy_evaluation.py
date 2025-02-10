import itertools
from test.fixtures import fresh_drive, medium_blocks
from typing import List

import torch

import xmimir as xmi
from rgnet.rl.optimality_utils import optimal_discounted_values, optimal_policy
from rgnet.rl.thundeRL.policy_evaluation import (
    PolicyEvaluationMessagePassing,
    build_mdp_graph_with_prob,
    mdp_graph_as_pyg_data,
)


def _placeholder_probs(space: xmi.XStateSpace):
    # Contains 0...(|Edges| - 1) as probabilities
    it = itertools.count()
    return [
        torch.Tensor(
            [next(it) for _ in range(space.forward_transition_count(state))],
        )
        for state in space
    ]


def test_build_mdp_graph(medium_blocks):
    space: xmi.XStateSpace = medium_blocks[0]
    nx_graph = build_mdp_graph_with_prob(space, _placeholder_probs(space))
    assert all(
        all(
            out_edge[0] == s.index and out_edge[1] == out_transition.target.index
            for out_edge, out_transition in zip(
                nx_graph.out_edges(nbunch=[i]), space.forward_transitions(s)
            )
        )
        for i, s in enumerate(space)
    )


def test_mdp_graph_as_pyg_data(medium_blocks):
    space: xmi.XStateSpace = medium_blocks[0]
    probs_list = _placeholder_probs(space)
    pyg_graph = mdp_graph_as_pyg_data(build_mdp_graph_with_prob(space, probs_list))
    # Check that the probabilities are stored in the edge_attr
    # Note that we cannot use positional comparison of probabilities stored, as the edges order is not guaranteed, i.e.
    # this is not a valid test:
    assert (pyg_graph.edge_attr[:, 0] == torch.cat(probs_list)).all()
    # Instead, we check that the probabilities are stored in the edge_attr cumulatively and each value is found
    # somewhere (hedge against different terms summing up to the correct value).
    assert pyg_graph.edge_attr[:, 0].sum() == torch.cat(probs_list).sum() and all(
        prob in pyg_graph.edge_attr[:, 0] for prob in torch.cat(probs_list)
    )


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
            (len(list(space.forward_transitions(state))),),
            dtype=torch.float,
        )
        probs[optimal_action_idx] = 1.0
        return probs

    optimal_policy_probabilities: List[torch.Tensor] = [
        optimal_probs(i, s) for (i, s) in enumerate(space)
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
        ).softmax(dim=-1)
        if i == one_before_goal_idx:
            probs = torch.zeros((nr_transitions,), dtype=torch.float)
            probs[1] = 1.0
        return probs

    probs_list = [faulty_probs(i, s) for (i, s) in enumerate(space)]

    graph = build_mdp_graph_with_prob(space, probs_list)
    graph_data = mdp_graph_as_pyg_data(graph)

    values = value_iteration_mp(graph_data)

    # The goal is never reached; therefore, the values for all states should go towards
    # the discounted infinite trajectory length, which is -1 / (1-gamma).
    expected_values = torch.full((len(space),), -1 / (1 - gamma))
    expected_values[goal_state.index] = 0.0
    assert torch.allclose(values, expected_values, atol=0.01)
    # We can never go beyond -1 / (1 gamma).
    #
    assert (values >= -10).all()
