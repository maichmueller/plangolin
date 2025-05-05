import itertools
from test.fixtures import (  # noqa: F401, F403
    fresh_flashdrive,
    medium_blocks,
    small_blocks,
)

import pytest
import torch

import xmimir as xmi
from rgnet.algorithms import optimal_policy
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.reward import UnitReward
from rgnet.utils import mdp_graph_as_pyg_data


def _placeholder_probs(space: xmi.XStateSpace):
    # Contains 0...(|Edges| - 1) as probabilities
    it = itertools.count()
    return tuple(
        torch.Tensor(
            [next(it) for _ in range(space.forward_transition_count(state))],
        )
        for state in space
    )


def test_build_mdp_graph(medium_blocks):
    space: xmi.XStateSpace = medium_blocks[0]
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=0.9),
    )
    env.reset()
    nx_graph = env.to_mdp_graph(0)
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
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=0.9),
    )
    env.reset()
    pyg_graph = mdp_graph_as_pyg_data(env.to_mdp_graph(0, probs_list))
    # Check that the probabilities are stored in the edge_attr
    # Note that we cannot use positional comparison of probabilities stored, as the edges order is not guaranteed, i.e.
    # this is not a valid test:
    assert (pyg_graph.edge_attr[:, 1] == torch.cat(probs_list)).all()
    # Instead, we check that the probabilities are stored in the edge_attr cumulatively and each value is found
    # somewhere (hedge against different terms summing up to the correct value).
    assert pyg_graph.edge_attr[:, 1].sum() == torch.cat(probs_list).sum() and all(
        prob in pyg_graph.edge_attr[:, 1] for prob in torch.cat(probs_list)
    )


@pytest.mark.parametrize(
    "problem",
    [
        "small_blocks",
        "medium_blocks",
    ],
)
def test_mdp_and_pyg_equivalence(problem, request):
    space, _, _ = request.getfixturevalue(problem)
    gamma = 0.9
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=gamma),
        reset=True,
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
    mdp_pyg_data = mdp_graph_as_pyg_data(graph)
    pyg_graph = env.to_pyg_data(0, optimal_policy_probabilities)

    assert torch.allclose(pyg_graph.edge_attr, mdp_pyg_data.edge_attr, 0.01)
    assert torch.allclose(pyg_graph.x, mdp_pyg_data.x, 0.01)
    assert torch.equal(pyg_graph.edge_index, mdp_pyg_data.edge_index)
    #
