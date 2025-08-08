from test.fixtures import embedding_mock, small_blocks  # noqa: F401

import mockito
import pytest
import torch.nn

from plangolin.rl.agents import ValueModule
from plangolin.rl.envs import ExpandedStateSpaceEnv, InitialStateReset


@pytest.mark.parametrize("embedding_size", [2])
def test_forward(embedding_mock, small_blocks, embedding_size):
    space, _, _ = small_blocks
    batch_size = 2

    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size([batch_size]),
        seed=42,
        reset_strategy=InitialStateReset(),
    )

    td = env.reset()
    assert all(
        len(transitions) == 2 for transitions in td[env.keys.transitions]
    ), "Test assumption failed expected small blocks to have two possible transitions."
    transitions = td[env.keys.transitions]
    expected_actions = [transitions[i][0] for i in range(batch_size)]

    value_net = torch.nn.Module()
    # We fix the values such that transitions[i][0] are best.
    # In small blocks the initial state has two possible transitions.
    value_net_return = torch.tensor(
        [4.0, 3.0, 2.0, 1.0], dtype=torch.float32, requires_grad=True
    ).unsqueeze(dim=-1)
    mockito.when(value_net).forward(...).thenReturn(value_net_return)

    value_module = ValueModule(embedding_mock, value_net)
    policy = value_module.as_td_module(env.keys.transitions, env.keys.action)

    td = policy(td)

    assert env.keys.action in td
    actions = td[env.keys.action]

    assert actions == expected_actions

    # once over whole batch at once
    mockito.verify(value_net, times=1).forward(...)
    mockito.verify(embedding_mock, times=1).forward(...)
