from test.fixtures import medium_blocks, small_blocks
from typing import List, Tuple

import pymimir as mi
import pytest
import torch
from tensordict import TensorDict

from rgnet.rl.envs import ExpandedStateSpaceEnv, MultiInstanceStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack


@pytest.fixture
def multi_instance_env(
    batch_size, small_blocks, medium_blocks, seed=42
) -> Tuple[mi.StateSpace, mi.StateSpace, MultiInstanceStateSpaceEnv]:
    space_small, _, _ = small_blocks
    space_medium, _, _ = medium_blocks

    return (
        space_small,
        space_medium,
        MultiInstanceStateSpaceEnv(
            spaces=[space_small, space_medium],
            batch_size=torch.Size((batch_size,)),
            seed=seed,
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_reset(multi_instance_env, batch_size):
    small_space, medium_space, environment = multi_instance_env

    td = environment.reset()
    expected_keys = [
        ExpandedStateSpaceEnv.default_keys.done,
        ExpandedStateSpaceEnv.default_keys.goals,
        ExpandedStateSpaceEnv.default_keys.state,
        ExpandedStateSpaceEnv.default_keys.terminated,
        ExpandedStateSpaceEnv.default_keys.transitions,
    ]
    assert td.sorted_keys == expected_keys

    if batch_size == 1:
        expected_states = [small_space.get_initial_state()]
    elif batch_size == 2:
        expected_states = [
            small_space.get_initial_state(),
            medium_space.get_initial_state(),
        ]
    else:
        expected_states = [
            small_space.get_initial_state(),
            medium_space.get_initial_state(),
            small_space.get_initial_state(),  # <- we only provided two spaces
        ]
    assert td[ExpandedStateSpaceEnv.default_keys.state] == expected_states

    predefined_td = TensorDict({}, batch_size=torch.Size([batch_size]))
    out = environment.reset(predefined_td)
    assert out is predefined_td
    assert out.sorted_keys == expected_keys


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_partial_reset(multi_instance_env, batch_size):
    small_space, medium_space, environment = multi_instance_env
    spaces = [small_space, medium_space]

    goal_state = small_space.get_goal_states()[0]
    transition_from_goal = small_space.get_forward_transitions(goal_state)[0]

    keys = environment.keys

    # Set the initial states such that (only) the first batch entry will be done
    batch_space_indices: List[int] = [0, 1, 0][:batch_size]
    initial_states = [spaces[idx].get_initial_state() for idx in batch_space_indices]
    initial_states[0] = goal_state
    td = environment.reset(states=initial_states)
    td = environment.rand_action(td)

    actions: List[mi.Transition] = td[keys.action]
    actions[0] = transition_from_goal
    td[keys.action] = as_non_tensor_stack(actions)

    tensordict, next_tensordict = environment.step_and_maybe_reset(td)
    # Only the first batch entry is done.
    assert (
        tensordict[("next", "done")].nonzero().view(-1) == torch.tensor([0, 0])
    ).all()
    expected_next_states = [a.target for a in tensordict["action"]]
    assert tensordict[("next", "state")] == expected_next_states

    expected_next_initial: mi.State
    if batch_size == 1 or batch_size == 3:
        expected_next_initial = medium_space.get_initial_state()
    elif batch_size == 2:
        expected_next_initial = small_space.get_initial_state()
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
    if batch_size == 1:
        return

    try:
        environment.rand_step(next_tensordict)
    except ValueError:
        pytest.fail("Internal state misconfiguration after partial reset")

    if batch_size == 2:
        expected_transitions = [
            small_space.get_forward_transitions(expected_next_states[0]),
            medium_space.get_forward_transitions(expected_next_states[1]),
        ]
    else:
        expected_transitions = [
            medium_space.get_forward_transitions(expected_next_states[0]),
            medium_space.get_forward_transitions(expected_next_states[1]),
            small_space.get_forward_transitions(expected_next_states[2]),
        ]

    # Assert that the chosen actions are sampled from the allowed transitions
    assert next_tensordict[keys.transitions] == expected_transitions
    next_random_actions = next_tensordict[keys.action]
    for idx in range(batch_size):
        assert next_random_actions[idx] in expected_transitions[idx]

    assert next_tensordict[("next", keys.state)] == [
        a.target for a in next_random_actions
    ]
