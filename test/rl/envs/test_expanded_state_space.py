from test.fixtures import *  # noqa: F401, F403
from typing import List

import pytest
import torch
from tensordict import TensorDict

from plangolin.algorithms import mdp_graph_as_pyg_data
from plangolin.rl.envs import ExpandedStateSpaceEnv
from plangolin.utils.misc import as_non_tensor_stack
from xmimir import XState, XStateSpace, XTransition

from .test_state_space_env import get_expected_root_keys


@pytest.mark.parametrize(
    "multi_instance_env",
    [
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=1),
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=2),
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=3),
    ],
    indirect=True,
)
def test_reset(multi_instance_env):
    environment = multi_instance_env
    small_space, medium_space = multi_instance_env._all_instances
    batch_size = environment.batch_size[0]
    td = environment.reset()
    expected_keys = get_expected_root_keys(environment)
    assert td.sorted_keys == expected_keys

    if batch_size == 1:
        expected_states = [small_space.initial_state]
    elif batch_size == 2:
        expected_states = [
            small_space.initial_state,
            medium_space.initial_state,
        ]
    else:
        expected_states = [
            small_space.initial_state,
            medium_space.initial_state,
            small_space.initial_state,  # <- we only provided two spaces
        ]
    assert td[ExpandedStateSpaceEnv.default_keys.state] == expected_states

    predefined_td = TensorDict({}, batch_size=torch.Size([batch_size]))
    out = environment.reset(predefined_td)
    assert out is predefined_td
    assert out.sorted_keys == expected_keys


@pytest.mark.parametrize(
    "multi_instance_env",
    [
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=1),
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=2),
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=3),
    ],
    indirect=True,
)
def test_partial_reset(multi_instance_env):
    environment = multi_instance_env
    small_space, medium_space = multi_instance_env._all_instances
    batch_size = environment.batch_size[0]
    spaces = [small_space, medium_space]

    goal_state = next(small_space.goal_states_iter())
    transition_from_goal = next(small_space.forward_transitions(goal_state))
    keys = environment.keys

    # Set the initial states such that (only) the first batch entry will be done
    batch_space_indices: List[int] = [0, 1, 0][:batch_size]
    initial_states = [spaces[idx].initial_state for idx in batch_space_indices]
    initial_states[0] = goal_state
    td = environment.reset(states=initial_states)

    # Assert that the active instances are set correctly
    assert td[keys.instance] == [spaces[idx] for idx in batch_space_indices]

    # Execute a random action but make sure that the first entry is done
    td = environment.rand_action(td)
    actions: List[XTransition] = td[keys.action]
    actions[0] = transition_from_goal
    td[keys.action] = as_non_tensor_stack(actions)

    tensordict, next_tensordict = environment.step_and_maybe_reset(td)
    # Only the first batch entry is done.
    assert (
        tensordict[("next", "done")].nonzero().view(-1) == torch.tensor([0, 0])
    ).all()
    expected_next_states = [a.target for a in tensordict["action"]]
    assert tensordict[("next", "state")] == expected_next_states

    expected_next_initial: XState
    if batch_size == 1 or batch_size == 3:
        expected_next_initial = medium_space.initial_state
    elif batch_size == 2:
        expected_next_initial = small_space.initial_state
    else:
        raise RuntimeError("Test was not design for batch_size > 3")

    expected_next_states[0] = expected_next_initial  # the first entry is reset
    assert next_tensordict[keys.state] == expected_next_states

    # If the partial reset was not handled then the non-reset entries will misbehave.
    # batch_size = 1 is trivial as it's just a full reset
    # For batch_size = 2 we have batch_size == len(all_instances) therefore every space is
    # replaced by itself.
    # For batch_size = 3 we have:
    # Before the partial reset happened the layout was:
    # 0: small-space
    # 1: medium-space
    # 2: small-space
    # We expect following layout because the first entry is done and has to be replaced.
    # 0: medium-space
    # 1: medium-space
    # 2: small-space
    # But if all batch-entries were replaced it would be:
    # 0: medium-space
    # 1: small-space
    # 2: medium-space

    # This of course can only happen for a batch_size > 1
    expected_active_instance: List[XStateSpace]
    if batch_size == 1:
        # Small instance was replaced with next which is medium
        assert next_tensordict[keys.instance] == [medium_space]
        return

    try:
        environment.rand_step(next_tensordict)
    except ValueError:
        pytest.fail("Internal state misconfiguration after partial reset")

    if batch_size == 2:
        expected_active_instance = [small_space, medium_space]
        expected_transitions = [
            list(small_space.forward_transitions(expected_next_states[0])),
            list(medium_space.forward_transitions(expected_next_states[1])),
        ]
    else:
        expected_active_instance = [medium_space, medium_space, small_space]
        expected_transitions = [
            list(medium_space.forward_transitions(expected_next_states[0])),
            list(medium_space.forward_transitions(expected_next_states[1])),
            list(small_space.forward_transitions(expected_next_states[2])),
        ]

    # Assert that the chosen actions are sampled from the allowed transitions
    assert next_tensordict[keys.transitions] == expected_transitions
    next_random_actions = next_tensordict[keys.action]
    for idx in range(batch_size):
        assert next_random_actions[idx] in expected_transitions[idx]

    # Assert that the active instances are set correctly
    assert next_tensordict[keys.instance] == expected_active_instance

    assert next_tensordict[("next", keys.state)] == [
        a.target for a in next_random_actions
    ]


@pytest.mark.parametrize(
    "multi_instance_env",
    [
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=1),
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=2),
        dict(spaces=["small_blocks", "medium_blocks"], batch_size=3),
        dict(
            spaces=[
                "small_blocks",
                "small_delivery_1_pkgs",
                "small_delivery_2_pkgs",
            ],
            batch_size=1,
        ),
        dict(
            spaces=[
                "small_delivery_1_pkgs",
                "small_blocks",
                "small_delivery_2_pkgs",
            ],
            batch_size=5,
        ),
    ],
    indirect=True,
)
def test_pyg_conversion(multi_instance_env):
    mdp_graphs = tuple(
        multi_instance_env.to_mdp_graph(i)
        for i in range(len(multi_instance_env._all_instances))
    )
    pyg_data_from_mdps = tuple(
        mdp_graph_as_pyg_data(mdp_graph) for mdp_graph in mdp_graphs
    )
    pyg_data_from_envs = tuple(
        multi_instance_env.to_pyg_data(i)
        for i in range(len(multi_instance_env._all_instances))
    )
    for pyg_data_from_env, pyg_data_from_mdp in zip(
        pyg_data_from_envs, pyg_data_from_mdps
    ):
        assert pyg_data_from_env.num_nodes == pyg_data_from_mdp.num_nodes
        assert pyg_data_from_env.num_edges == pyg_data_from_mdp.num_edges
        assert torch.all(pyg_data_from_env.x == pyg_data_from_mdp.x)
        assert torch.all(pyg_data_from_env.edge_index == pyg_data_from_mdp.edge_index)
        assert torch.allclose(pyg_data_from_env.edge_attr, pyg_data_from_mdp.edge_attr)
        assert torch.all(pyg_data_from_env.goals == pyg_data_from_mdp.goals)
        assert pyg_data_from_env.gamma == pyg_data_from_mdp.gamma
