from test.fixtures import small_blocks
from typing import Iterable

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


def _test_rollout_soundness(
    space,
    rollout,
    batch_size,
    rollout_length,
    initial_state,
    expected_keys: Iterable = None,
):

    expected_root_keys = expected_keys or [
        ExpandedStateSpaceEnv.default_keys.action,
        ExpandedStateSpaceEnv.default_keys.done,
        ExpandedStateSpaceEnv.default_keys.goals,
        ExpandedStateSpaceEnv.default_keys.state,
        ExpandedStateSpaceEnv.default_keys.terminated,
        ExpandedStateSpaceEnv.default_keys.transitions,
    ]
    assert "next" in rollout.keys()
    expected_root_keys = sorted(expected_root_keys)

    actual_keys = rollout.sorted_keys
    actual_keys.remove("next")
    assert actual_keys == expected_root_keys

    for key in expected_root_keys:
        val = rollout.get(key)
        if isinstance(val, torch.Tensor):
            assert val.shape[0] == batch_size and val.shape[1] == rollout_length
        else:
            assert isinstance(val, NonTensorStack)
            assert val.batch_size == (batch_size, rollout_length)

    keys = ExpandedStateSpaceEnv.default_keys
    # Test that the rollout makes sense
    batched_curr_state = rollout[keys.state]
    batched_transitions = rollout[keys.transitions]
    batched_actions = rollout[keys.action]
    for time_step in range(0, rollout_length):
        for batch_idx in range(0, batch_size):
            current_state = batched_curr_state[batch_idx][time_step]
            transitions = space.get_forward_transitions(current_state)
            assert batched_transitions[batch_idx][time_step] == transitions
            assert batched_actions[batch_idx][time_step] in transitions
            if time_step == rollout_length - 1:
                continue
            # If we transition into a done-state, torchrl will reset directly and
            # replace the done state with a reset state
            if space.is_goal_state(rollout[keys.action][batch_idx][time_step].target):
                assert rollout["next", keys.done][batch_idx][time_step]
                assert (
                    rollout["next", keys.state][batch_idx][time_step]
                    == rollout[keys.action][batch_idx][time_step].target
                )
                assert batched_curr_state[batch_idx][time_step + 1] == initial_state
            else:
                assert (
                    batched_curr_state[batch_idx][time_step + 1]
                    == rollout[keys.action][batch_idx][time_step].target
                )


@pytest.mark.parametrize("batch_size", [1, 2, 3])
# for seed=0 a partial reset is necessary because a goal is encountered
@pytest.mark.parametrize("seed", [0, 42])
def test_rollout_random(batch_size, small_blocks, seed):
    space, environment = create_state_space(batch_size, small_blocks, seed)
    rollout_length = 5
    rollout = environment.rollout(
        rollout_length, break_when_any_done=False, auto_reset=True
    )
    _test_rollout_soundness(
        space, rollout, batch_size, rollout_length, environment._initial_state
    )


@pytest.mark.parametrize("batch_size", [2, 3])
def test_rollout_reset(small_blocks, batch_size):
    """Test the (partial) reset of the environment.
    If at batch_index an action results in a done state we expect that
        1. td['next','done'] is true
        2. td['next', 'state'] is the target of the action
        3. td['state'] at the next time step is the initial-state
        All three conditions should only apply for the specific batch_index and not the
        other batch entries!
    """
    space, environment = create_state_space(batch_size, small_blocks)

    initial_state = space.get_initial_state()
    goal_state = space.get_goal_states()[0]
    initial_transitions = space.get_forward_transitions(initial_state)
    non_final_transition = next(
        t for t in initial_transitions if not space.is_goal_state(t.target)
    )
    final_transition = space.get_backward_transitions(goal_state)[0]
    one_before_goal_state = final_transition.source

    if batch_size == 1:
        one_before_goal = [one_before_goal_state] * batch_size
        actions = [final_transition] * batch_size
    else:
        one_before_goal = [one_before_goal_state] + [initial_state] * (batch_size - 1)
        actions = [final_transition] + [non_final_transition] * (batch_size - 1)

    td = environment.reset()
    keys = ExpandedStateSpaceEnv.default_keys
    td[keys.state] = as_non_tensor_stack(one_before_goal)
    td[keys.action] = as_non_tensor_stack(actions)
    # step_and_maybe_reset returns the input td and the t+1 tensordict
    # the input td will have a next key, the t+1 tensordict will be partially be set back
    out_td, reset_td = environment.step_and_maybe_reset(td)

    assert out_td is td  # should return the same tensordict as input
    assert "next" in td
    assert td["next", keys.state] == [t.target for t in td[keys.action]]
    assert td["next", "done"][0].item()  # first batch entry is done
    if batch_size > 1:
        assert not td["next", "done"][1:].any()  # no other batch entry is done
    assert not reset_td["done"].view(-1).any()
    assert reset_td[keys.state][0] == initial_state  # reset occurred for first entry
    if batch_size > 1:  # no other entries were reset
        assert not any(state == initial_state for state in reset_td[keys.state][1:])


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
