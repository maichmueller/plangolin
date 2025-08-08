import math
from test.fixtures import small_blocks  # noqa: F401

import mockito
import pytest
import torch

from plangolin.rl.agents import EGreedyModule, EpsilonAnnealing
from plangolin.rl.envs import ExpandedStateSpaceEnv
from plangolin.utils.misc import as_non_tensor_stack


def test_step_epsilon():
    annealing = EpsilonAnnealing(epsilon_init=0.5, epsilon_end=0.1, annealing_steps=5)
    assert annealing.epsilon == 0.5
    annealing.step_epsilon()
    assert math.isclose(annealing.epsilon, 0.5 - (0.5 - 0.1) / 5, abs_tol=0.001)
    annealing.step_epsilon()
    annealing.step_epsilon()
    annealing.step_epsilon()
    assert not math.isclose(annealing.epsilon, 0.1, abs_tol=0.001)
    annealing.step_epsilon()
    assert math.isclose(annealing.epsilon, 0.1, abs_tol=0.001)


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
    initial_states = space[0:2]
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
