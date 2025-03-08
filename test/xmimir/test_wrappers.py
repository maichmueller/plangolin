from collections import deque
from test.fixtures import large_blocks, medium_blocks, small_blocks  # noqa: F401

import pymimir
import pytest
import torch
from tensordict.tensorclass import NonTensorData
from torchrl.envs.transforms import RenameTransform

import xmimir as xmi
from rgnet.rl.envs import ExpandedStateSpaceEnv
from xmimir import XProblem
from xmimir.wrappers import StateLabel, XSearchResult


@pytest.mark.parametrize("space_fixture", ["small_blocks", "medium_blocks"])
def test_xstate_label(space_fixture, request):
    space, _, problem = request.getfixturevalue(space_fixture)
    initial_state = space.initial_state
    assert isinstance(initial_state, xmi.XState)
    assert initial_state.label == xmi.StateLabel.initial

    goal_states = list(iter(space.goal_states_iter()))
    assert all(
        isinstance(goal_state, xmi.XState) and goal_state.label == xmi.StateLabel.goal
        for goal_state in goal_states
    )

    deadend_states = list(iter(space.deadend_states_iter()))
    assert all(
        isinstance(deadend_state, xmi.XState)
        and deadend_state.label == xmi.StateLabel.deadend
        for deadend_state in deadend_states
    )

    # we trust that successor_generator is using the same StateRepository of the state space!
    succ_gen = space.successor_generator
    assert isinstance(succ_gen, xmi.XSuccessorGenerator)
    initial_state = succ_gen.initial_state
    assert isinstance(initial_state, xmi.XState)
    assert initial_state.label == xmi.StateLabel.initial

    all_states = set(space)
    n_states = len(all_states)
    seen = {initial_state}
    queue = deque([initial_state])
    while len(seen) < n_states:
        current = queue.popleft()
        for action, next_state in succ_gen.successors(current):
            assert isinstance(action, xmi.XAction)
            assert isinstance(next_state, xmi.XState)
            assert next_state in all_states
            if next_state not in seen:
                assert next_state.label == xmi.StateLabel.unknown
                if (
                    next_state.is_goal()
                ):  # the check should also update the label if it turns out to be a goal
                    assert next_state.label == xmi.StateLabel.goal
            seen.add(next_state)
            queue.append(next_state)


@pytest.mark.parametrize("space_fixture", ["small_blocks", "medium_blocks"])
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
# @pytest.mark.parametrize("batch_size", [5])
def test_state_label_in_env(space_fixture, batch_size, request):
    space, _, problem = request.getfixturevalue(space_fixture)
    result = space.breadth_first_search(space.initial_state)
    # result.plan
    successor_generator = space.successor_generator
    transitions = []
    source = result.start
    for action in result.plan:
        target = successor_generator.successor(source, action)
        transitions.append(xmi.XTransition.make_hollow(source, target, action))
        source = target
    # print(*map(str, result.plan), sep="\n")
    env = ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)
    td = env.reset()
    step_count = 0
    while step_count < len(result):
        for s in td[env.keys.state]:
            ...
            assert s.label == (
                StateLabel.unknown if step_count > 0 else StateLabel.initial
            )
        td[env.keys.action] = NonTensorData([transitions[step_count]] * batch_size)
        td = env.step(td)["next"]
        step_count += 1
