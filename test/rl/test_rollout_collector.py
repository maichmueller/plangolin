import itertools
from collections import deque
from math import ceil
from test.fixtures import *  # noqa: F401, F403
from typing import List

import mockito
import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

import xmimir as xmi
from plangolin.rl.envs.expanded_state_space_env import (
    ExpandedStateSpaceEnv,
    InitialStateReset,
    IteratingReset,
    MultiInstanceStateSpaceEnv,
    WeightedRandomReset,
)
from plangolin.rl.envs.planning_env import PlanningEnvironment, RoundRobinReplacement
from plangolin.rl.rollout_collector import RolloutCollector, build_from_spaces
from plangolin.utils.misc import as_non_tensor_stack, tolist
from xmimir import XStateSpace


def test_build_from_spaces_multiple(small_blocks, medium_blocks, batch_size=5):
    s_space, _, _ = small_blocks
    m_space, _, _ = medium_blocks
    total_states = len(s_space) + len(m_space)
    assert total_states == 130 or total_states % batch_size == 0
    collector = build_from_spaces(
        [s_space, m_space], batch_size=batch_size, rollout_length=1
    )
    assert isinstance(collector.env, MultiInstanceStateSpaceEnv)
    assert isinstance(collector.env.reset_strategy, IteratingReset)
    assert isinstance(collector.env._instance_replacement_strategy, WeightedRandomReset)

    batch_counter = 0
    for batch in collector:
        batch_counter += 1
        # We have a rollout length of 1
        assert batch.batch_size == (batch_size, 1)
        assert batch.names == [None, "time"]
        if batch_counter == 0:
            states = batch[PlanningEnvironment.default_keys.state]
            instances: List[XStateSpace] = batch[
                PlanningEnvironment.default_keys.instance
            ]
            assert len(states) == len(instances) == batch_size
            assert states[0] == instances[0].get_state(0)

    assert batch_counter == ceil(total_states / batch_size)


@pytest.mark.parametrize(
    "blocks",
    ["small_blocks", "medium_blocks"],
)
def test_build_from_spaces_single(request, blocks, batch_size=5):
    space = request.getfixturevalue(blocks)[0]
    collector = build_from_spaces(space, batch_size=batch_size, rollout_length=1)
    assert len(space) % batch_size == 0, "Sanity check that test setup is correct."

    assert isinstance(collector.env, ExpandedStateSpaceEnv)
    assert isinstance(collector.env.reset_strategy, IteratingReset)
    assert isinstance(collector.env._instance_replacement_strategy, WeightedRandomReset)

    batch_counter = 0
    batches = []
    for batch in collector:
        batch_counter += 1
        batches.append(batch)
    if len(space) == batch_size:
        assert batch_counter == 1
        # batch_size = s_space.get_num_vertices() + IteratingReset()
        # As we have a rollout length of 1 each batch entry is a list of length one
        states_flattened = list(
            itertools.chain.from_iterable(
                batches[-1][PlanningEnvironment.default_keys.state]
            )
        )
        assert states_flattened == list(space.states_iter())
    else:
        assert batch_counter == ceil(len(space) / batch_size)
        all_states = list(space.states_iter())
        all_encountered_states = []
        for batch in batches:
            states = batch[PlanningEnvironment.default_keys.state]
            all_encountered_states.extend(states)
        all_encountered_states_flattened = list(
            itertools.chain.from_iterable(all_encountered_states)
        )
        # As we only have one instance and use IteratingReset we should encounter all states as they are layed out in the space.
        assert all_encountered_states_flattened == all_states


@pytest.mark.parametrize(
    "expanded_state_space_env", [["small_blocks", 5]], indirect=True
)
def test_with_policy(expanded_state_space_env):
    """Test that the policy is actually used to collect the rollouts."""
    env = expanded_state_space_env

    def policy(transitions):
        transitions: List[List[xmi.XTransition]] = tolist(transitions)
        actions = [transitions[i][0] for i in range(len(transitions))]
        return as_non_tensor_stack(actions)

    tdm = TensorDictModule(
        module=policy,
        in_keys=[PlanningEnvironment.default_keys.transitions],
        out_keys=[PlanningEnvironment.default_keys.action],
    )
    mockito.spy2(tdm.forward)

    collector = RolloutCollector(
        environment=env,
        policy=tdm,
        num_batches=3,
        rollout_length=1,
    )
    # we use deque to exhaust the collector instead of e.g. list, since `list` will ask for __len__ of the collector,
    # which fails for collectors that do not set the `frames_per_batch` and `total_frames` (torchrl v0.7).
    # Also, deque consumes at C-speed, while list assigns outputs to variables within python
    # (in general more relevant, here not so much).
    deque(collector, maxlen=0)  # exhaust the collector

    # The policy gets called once for each batch and time step
    mockito.verify(tdm, times=3).forward(...)


@pytest.mark.parametrize("rollout_length", [1, 2])
def test_reset_after_each_batch(
    medium_blocks, rollout_length, batch_size=5, num_batches=5
):
    """Test that all instances are replaced after each batch, when setting reset_after_each_batch."""
    space = medium_blocks[0]
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((batch_size,)),
        # It is important that we use initial state reset as we want to avoid reaching
        # terminating states which would also trigger instance replacements.
        reset_strategy=InitialStateReset(),
    )

    replacement_strategy_mock = mockito.mock(spec=RoundRobinReplacement)
    mockito.when(replacement_strategy_mock).__call__(...).thenReturn(space)

    env._instance_replacement_strategy = replacement_strategy_mock
    assert isinstance(env._instance_replacement_strategy, RoundRobinReplacement)
    collector = RolloutCollector(
        environment=env,
        policy=None,
        num_batches=num_batches,
        rollout_length=rollout_length,
    )
    # exhaust the collector
    for _ in collector:
        pass

    # We expect that each instance is replaced for each batch
    # we have batch_size * num_batches replacements
    mockito.verify(replacement_strategy_mock, times=num_batches * batch_size).__call__(
        ...
    )
    for batch_entry in range(batch_size):
        mockito.verify(replacement_strategy_mock, times=num_batches).__call__(
            batch_entry
        )


@pytest.mark.parametrize(
    "expanded_state_space_env", [["small_blocks", 5]], indirect=True
)
def test_reset(expanded_state_space_env):
    """Test that we can reiterate the collector after a reset."""
    env = expanded_state_space_env
    env.make_replacement_strategy(RoundRobinReplacement)

    collector = RolloutCollector(
        environment=env,
        policy=None,
        num_batches=1,
        rollout_length=1,
    )

    batch = next(collector)
    assert isinstance(batch, TensorDict)
    try:
        next(collector)
        pytest.fail()
    except StopIteration:
        pass
    except Exception as e:
        pytest.fail(f"Unexpected exception {e}")

    collector.reset()
    batch_after_reset = next(collector)
    assert isinstance(batch_after_reset, TensorDict)
