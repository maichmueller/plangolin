from test.fixtures import medium_blocks  # noqa: F401

import pytest
import torch

from plangolin.algorithms import OptimalPolicyMP, mdp_graph_as_pyg_data, optimal_policy
from plangolin.rl.envs import ExpandedStateSpaceEnv
from plangolin.rl.reward import UnitReward


@pytest.mark.parametrize(
    "env_representation",
    [
        "pyg",
        "networkx",
    ],
)
def test_optimal_policy_mp_returns_local_argmax_indices(
    env_representation, medium_blocks
):
    # Arrange
    space, _, _ = medium_blocks
    gamma = 0.9

    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=gamma),
        reset=True,
    )

    # crude optimal policy: state -> {local action indices}
    opt = optimal_policy(space)

    # Build one‑hot per-state probabilities aligned with forward_transitions order
    def one_hot_probs(i, state):
        actions = list(space.forward_transitions(state))
        if not actions:
            return torch.zeros((0,), dtype=torch.float)
        idx = next(iter(opt[i]))  # pick a single optimal action if set has >1
        probs = torch.zeros((len(actions),), dtype=torch.float)
        probs[idx] = 1.0
        return probs

    optimal_policy_probabilities: tuple[torch.Tensor, ...] = tuple(
        one_hot_probs(i, s) for (i, s) in enumerate(space)
    )

    if env_representation == "networkx":
        graph_nx = env.to_mdp_graph(0, optimal_policy_probabilities)
        graph = mdp_graph_as_pyg_data(graph_nx)
    else:
        graph = env.to_pyg_data(0, optimal_policy_probabilities)

    # Act: single greedy backup to extract argmax per state
    mp = OptimalPolicyMP(gamma=gamma, num_iterations=100, difference_threshold=None)
    out = mp(graph)

    # Prefer cached attribute if present; otherwise, use returned tensor
    attr_name = OptimalPolicyMP.default_attr_name
    pred = getattr(graph, attr_name) if hasattr(graph, attr_name) else out
    pred = pred.squeeze()

    # Expected local indices (or -1 for terminal states)
    expected = []
    for i, s in enumerate(space):
        deg = len(list(space.forward_transitions(s)))
        expected.append(-1 if deg == 0 else next(iter(opt[i])))
    expected = torch.tensor(expected, dtype=torch.long)

    # Assert
    assert pred.shape == expected.shape
    assert torch.equal(pred, expected), (
        f"OptimalPolicyMP local argmax indices differ.\n"
        f"pred:     {pred.tolist()}\n"
        f"expected: {expected.tolist()}"
    )
