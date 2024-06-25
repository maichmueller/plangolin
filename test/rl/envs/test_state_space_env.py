from test.fixtures import small_blocks

import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack


def create_state_space(batch_size, blocks, seed=42):
    space, domain, _ = blocks

    environment = ExpandedStateSpaceEnv(
        space=space, batch_size=torch.Size([batch_size]), seed=seed
    )
    return space, environment


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_reset(batch_size, small_blocks):
    space, environment = create_state_space(batch_size, small_blocks)
    td = environment.reset()
    expected_keys = [
        ExpandedStateSpaceEnv.default_keys.done,
        ExpandedStateSpaceEnv.default_keys.goals,
        ExpandedStateSpaceEnv.default_keys.state,
        ExpandedStateSpaceEnv.default_keys.terminated,
        ExpandedStateSpaceEnv.default_keys.transitions,
    ]

    assert td.sorted_keys == expected_keys

    predefined_td = TensorDict({}, batch_size=torch.Size([batch_size]))
    out = environment.reset(predefined_td)
    assert out is predefined_td
    assert out.sorted_keys == expected_keys


@pytest.mark.parametrize("batch_size", [1, 2, 3])
# for seed=0 a partial reset is necessary because a goal is encountered
@pytest.mark.parametrize("seed", [0, 42])
def test_rollout_random(batch_size, small_blocks, seed):
    space, environment = create_state_space(batch_size, small_blocks, seed)
    rollout_length = 5
    rollout = environment.rollout(rollout_length, break_when_any_done=False)
    keys = ExpandedStateSpaceEnv.default_keys
    assert "next" in rollout.keys()
    expected_root_keys = [
        keys.action,
        keys.done,
        keys.goals,
        keys.state,
        keys.terminated,
        keys.transitions,
    ]
    for key in expected_root_keys:
        assert key in rollout
        val = rollout.get(key)
        if isinstance(val, torch.Tensor):
            assert val.shape == (batch_size, rollout_length, 1)
        else:
            assert isinstance(val, NonTensorStack)
            assert val.batch_size == (batch_size, rollout_length)

    # Test that the rollout makes sense
    batched_curr_state = rollout[keys.state]
    batched_transitions = rollout[keys.transitions]
    batched_actions = rollout[keys.action]
    for time_step in range(0, rollout_length):
        any_done = rollout[keys.done][:, time_step, :].any()
        if any_done:  # If any is done the whole batch is reset
            continue
        for batch_idx in range(0, batch_size):
            current_state = batched_curr_state[batch_idx][time_step]
            transitions = space.get_forward_transitions(current_state)
            assert batched_transitions[batch_idx][time_step] == transitions
            assert batched_actions[batch_idx][time_step] in transitions
            if time_step != rollout_length - 1:
                assert (
                    batched_curr_state[batch_idx][time_step + 1]
                    == rollout[keys.action][batch_idx][time_step].target
                )


# @pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_rollout_reset(small_blocks):
    batch_size = 1
    space, environment = create_state_space(batch_size, small_blocks)
    td = environment.reset()
    # set
    keys = ExpandedStateSpaceEnv.default_keys
    goal_state = space.get_goal_states()[0]
    final_transition = space.get_backward_transitions(goal_state)[0]
    previous = final_transition.source
    environment.reset()
    one_before_goal = [previous] * batch_size
    td[keys.state] = as_non_tensor_stack(one_before_goal)
    environment.set_state(states=one_before_goal)
    td[keys.action] = as_non_tensor_stack([final_transition] * batch_size)
    _, reset_td = environment.step_and_maybe_reset(td)
    # We expect that a reset occurred and test it by checking whether the current state
    # is the _initial_state used in the _reset() logic.
    # NOTE _reset() logic might change in the future!
    assert all(s == environment._initial_state for s in reset_td[keys.state])


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_step_mdp(batch_size, small_blocks):
    _, environment = create_state_space(batch_size, small_blocks)
    td = environment.reset()
    td = environment.rand_step(td)
    next_td = environment._step_mdp(td)

    expected_keys = [
        ExpandedStateSpaceEnv.default_keys.action,
        ExpandedStateSpaceEnv.default_keys.done,
        ExpandedStateSpaceEnv.default_keys.goals,
        ExpandedStateSpaceEnv.default_keys.state,
        ExpandedStateSpaceEnv.default_keys.terminated,
        ExpandedStateSpaceEnv.default_keys.transitions,
    ]
    assert next_td.sorted_keys == expected_keys

    for tensor_key in (
        ExpandedStateSpaceEnv.default_keys.done,
        ExpandedStateSpaceEnv.default_keys.terminated,
    ):
        assert next_td[tensor_key] is td[("next", tensor_key)]

    for non_tensor_key in (
        ExpandedStateSpaceEnv.default_keys.goals,
        ExpandedStateSpaceEnv.default_keys.state,
        ExpandedStateSpaceEnv.default_keys.transitions,
    ):
        assert next_td[non_tensor_key] == td[("next", non_tensor_key)]
