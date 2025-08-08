from test.fixtures import *  # noqa: F401, F403
from typing import Iterable, List

import pytest
import torch
from tensordict import NestedKey, NonTensorStack, TensorDict

from plangolin.rl.envs import ExpandedStateSpaceEnv
from plangolin.rl.envs.planning_env import PlanningEnvironment


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
    expected_keys = get_expected_root_keys(environment)

    assert td.sorted_keys == expected_keys

    predefined_td = TensorDict({}, batch_size=torch.Size([batch_size]))
    out = environment.reset(predefined_td)
    assert out is predefined_td
    assert out.sorted_keys == expected_keys


def get_expected_root_keys(
    environment: PlanningEnvironment, with_action: bool = False
) -> List[NestedKey]:
    all_keys = (
        environment.done_keys
        + list(environment.full_observation_spec.keys(True, True))
        + list(environment.full_state_spec.keys(True, True))
    )
    if with_action:
        all_keys.append(environment.keys.action)
    return sorted(all_keys)


def get_expected_next_keys(
    environment: PlanningEnvironment, with_action: bool = False
) -> List[NestedKey]:
    return sorted(
        get_expected_root_keys(environment, with_action) + environment.reward_keys
    )


def _test_rollout_soundness(
    space,
    rollout,
    batch_size,
    rollout_length,
    initial_state,
    expected_root_keys: Iterable[NestedKey] = None,
    set_truncated: bool = False,
):
    assert "next" in rollout.keys()
    expected_root_keys = sorted(expected_root_keys)

    actual_keys = rollout.sorted_keys
    actual_keys.remove("next")
    assert actual_keys == expected_root_keys

    def validate_shape(value):
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == batch_size and value.shape[1] == rollout_length
        elif isinstance(value, TensorDict):
            for nested in value.values():
                validate_shape(nested)
        else:
            assert isinstance(value, NonTensorStack)
            assert value.batch_size == (batch_size, rollout_length)

    for key in expected_root_keys:
        val = rollout.get(key)
        validate_shape(val)

    keys = ExpandedStateSpaceEnv.default_keys
    # Test that the rollout makes sense
    batched_curr_state = rollout[keys.state]
    batched_transitions = rollout[keys.transitions]
    batched_actions = rollout[keys.action]
    for time_step in range(0, rollout_length):
        for batch_idx in range(0, batch_size):
            current_state = batched_curr_state[batch_idx][time_step]
            transitions = list(space.forward_transitions(current_state))
            non_goal_transitions_filter = lambda t: not t.source.is_goal()
            assert list(
                filter(
                    non_goal_transitions_filter,
                    batched_transitions[batch_idx][time_step],
                )
            ) == list(filter(non_goal_transitions_filter, transitions))
            if not (
                transition := batched_actions[batch_idx][time_step]
            ).source.is_goal():
                assert transition in transitions
            if time_step == rollout_length - 1:
                if set_truncated:
                    assert rollout["next", keys.done][batch_idx][time_step]
                    assert rollout["next", keys.truncated][batch_idx][time_step]
                continue
            # If we transition into a done-state, torchrl will reset directly and
            # replace the done state with a reset state
            # The environment will set terminated if a goal state is left!
            available_transitions: List = rollout[keys.transitions][batch_idx][
                time_step
            ]
            is_dead_end = (
                len(available_transitions) == 1
                and available_transitions[0].action is None
            )
            if (
                space.is_goal(rollout[keys.action][batch_idx][time_step].source)
                or is_dead_end
            ):
                assert rollout["next", keys.done][batch_idx][time_step]
                assert rollout["next", keys.terminated][batch_idx][time_step]
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
@pytest.mark.parametrize("set_truncated", [True, False])
def test_rollout_random(batch_size, small_blocks, seed, set_truncated):
    space, environment = create_state_space(batch_size, small_blocks, seed)
    rollout_length = 5
    rollout = environment.rollout(
        rollout_length,
        break_when_any_done=False,
        auto_reset=True,
        set_truncated=set_truncated,
    )
    _test_rollout_soundness(
        space,
        rollout,
        batch_size,
        rollout_length,
        space.initial_state,
        expected_root_keys=get_expected_root_keys(environment, with_action=True),
        set_truncated=set_truncated,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_rollout_reset(small_blocks, batch_size):
    """Test the (partial) reset of the environment.
    If at batch_index an action was taken from a goal state we expect:
        1. td['next','done'] is true
        2. td['next', 'state'] is the target of the action
        3. td['state'] at the next time step is the initial-state
        All three conditions should only apply for the specific batch_index and not the
        other batch entries!
    """
    space, environment = create_state_space(batch_size, small_blocks)

    initial_state = space.initial_state
    goal_state = next(space.goal_states_iter())

    initial_states = [initial_state] * batch_size
    initial_states[0] = goal_state

    td = environment.reset(states=initial_states)
    keys = environment.keys
    environment.rand_action(td)  # every action from the goal will trigger done
    # step_and_maybe_reset returns the input td and the t+1 tensordict
    # the input td will have a next key, the t+1 tensordict will be partially be set back
    out_td, reset_td = environment.step_and_maybe_reset(td)

    assert out_td is td  # should return the same tensordict as input
    assert "next" in td
    assert td["next", keys.state] == [t.target for t in td[keys.action]]
    assert td["next", "done"][0].item()  # first batch entry is done
    if batch_size > 1:
        assert not td["next", "done"][1:].any()  # no other batch entry is done
    # Test the tensordict for t+1
    assert not reset_td["done"].view(-1).any()  # none are done
    assert reset_td[keys.state][0] == initial_state  # reset occurred for first entry
    if batch_size > 1:  # no other entries were reset
        assert not any(state == initial_state for state in reset_td[keys.state][1:])


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_step_mdp(batch_size, small_blocks):
    _, environment = create_state_space(batch_size, small_blocks)
    td = environment.reset()
    td = environment.rand_step(td)
    next_td = environment._step_mdp(td)

    expected_keys = get_expected_root_keys(environment, with_action=True)
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
