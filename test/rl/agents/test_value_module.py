from test.fixtures import embedding_mock, small_blocks

import mockito
import pytest
import torch.nn

from rgnet.rl.agents import ValueModule
from rgnet.rl.envs import ExpandedStateSpaceEnv, InitialStateReset


@pytest.mark.parametrize("hidden_size", [2])
def test_forward(embedding_mock, small_blocks, hidden_size):
    space, _, _ = small_blocks
    batch_size = 2
    value_net = torch.nn.Module()

    # We fix the values such that transitions[i][0] are best.
    # In small blocks the initial state has two possible transitions.
    mockito.when(value_net).forward(...).thenReturn(
        torch.tensor([2.0, 1.0], requires_grad=True)
    )

    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size([batch_size]),
        seed=42,
        reset_strategy=InitialStateReset(),
    )

    value_module = ValueModule(embedding_mock, value_net)
    policy = value_module.as_td_module(env.keys.transitions, env.keys.action)

    td = env.reset()
    transitions = td[env.keys.transitions]
    expected_actions = [transitions[i][0] for i in range(batch_size)]

    td = policy(td)

    assert env.keys.action in td
    actions = td[env.keys.action]

    assert actions == expected_actions

    # once for every successors of the batch a.k.a batch_size
    mockito.verify(value_net, times=2).forward(...)
    mockito.verify(embedding_mock, times=1).forward(...)
