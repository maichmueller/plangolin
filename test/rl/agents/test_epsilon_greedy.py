from test.fixtures import embedding_mock, small_blocks

import mockito
import pytest
import torch

from rgnet.rl.agents import EGreedyModule, EpsilonAnnealing, ValueModule
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack


@pytest.mark.parametrize("batch_size", [2])
def test_forward(small_blocks, batch_size):
    space, _, _ = small_blocks
    env = ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)
    annealing = mockito.mock(
        {"step_epsilon": lambda *args: None}, spec=EpsilonAnnealing
    )
    annealing.epsilon = 1.0

    # Sample the first action
    mockito.when(torch).randint(...).thenReturn(torch.tensor([0]))

    egreedy = EGreedyModule(
        epsilon_annealing=annealing,
        transitions_key=env.keys.transitions,
        actions_key=env.keys.action,
    )
    assert batch_size == 2
    # select non-goal states with 2 transitions
    initial_states = space.get_states()[0:2]
    td = env.reset(states=initial_states)
    td[env.keys.action] = as_non_tensor_stack(
        [ts[1] for ts in td[env.keys.transitions]]
    )

    out = egreedy(td)

    # Epsilon is 1.0 so every action should be re-sampled
    # torch.randint is fixed to 0 so the new actions should be the first transition
    expected_transitions = [ts[0] for ts in td[env.keys.transitions]]
    assert out[env.keys.action] == expected_transitions

    # Only one step per batch
    mockito.verify(annealing, times=1).step_epsilon()
