from test.fixtures import small_blocks
from typing import List, Tuple

import pytest
import torch

import xmimir as xmi
from rgnet.rl.envs.planning_env import InstanceType, PlanningEnvironment
from xmimir import XLiteral, XState, XStateSpace


class DeadEndGoalEnv(PlanningEnvironment[XStateSpace]):
    """This environment will return a goal state that is a dead end."""

    def __init__(
        self,
        space,
        batch_size,
        custom_dead_end_reward,
        is_dead_end=True,
        seed=None,
        is_goal=True,
    ):
        super().__init__(
            all_instances=[space],
            batch_size=batch_size,
            seed=seed,
            device="cpu",
            custom_dead_end_reward=custom_dead_end_reward,
        )
        self.is_dead_end = is_dead_end
        self.is_goal_state = is_goal

    def transitions_for(
        self, active_instance: XStateSpace, state: XState
    ) -> List[xmi.XTransition]:
        return (
            [] if self.is_dead_end else list(active_instance.forward_transitions(state))
        )

    def initial_for(
        self, active_instance: XStateSpace
    ) -> Tuple[XState, List[XLiteral]]:
        return active_instance.initial_state(), list(active_instance.problem.goal())

    def is_goal(self, active_instance: XStateSpace, state: XState) -> bool:
        return self.is_goal_state


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("is_dead_end", [True, False])
@pytest.mark.parametrize("is_goal", [True, False])
def test_dead_end_transition(small_blocks, batch_size, is_dead_end, is_goal):
    """Tests the environment in regard of dead end states and goals states.
    The rollout should be done if any action was chosen from a dead end state or goal state.
    There should only be one action available from a dead end state which leads to itself.
    The reward depends on both conditions, precedence: default < dead end < goal"""
    space, _, _ = small_blocks
    custom_dead_end_reward = -100.0
    env = DeadEndGoalEnv(
        space=space,
        batch_size=torch.Size([batch_size]),
        seed=42,
        custom_dead_end_reward=custom_dead_end_reward,
        is_dead_end=is_dead_end,
        is_goal=is_goal,
    )

    td = env.reset()
    assert td[env.keys.goals] == [list(space.problem.goal())] * batch_size
    batched_transitions: List[List[xmi.XTransition]] = td[env.keys.transitions]
    if is_dead_end:
        assert all(len(ts) == 1 for ts in batched_transitions)
        assert all(ts[0].action is None for ts in batched_transitions)
        assert all(ts[0].source == ts[0].target for ts in batched_transitions)
        # There is only one action available
    td = env.rand_action(td)
    td, next_td = env.step_and_maybe_reset(td)
    if is_dead_end or is_goal:
        assert td[("next", env.keys.done)].all()
        assert td[("next", env.keys.terminated)].all()
        assert next_td[env.keys.state] == [space.initial_state()] * batch_size
        if is_dead_end:  # we expect to stay in the same state
            assert td[("next", env.keys.state)] == td[env.keys.state]
        else:
            assert td[("next", env.keys.state)] == [
                t.target for t in td[env.keys.action]
            ]

    reward: torch.Tensor = td[("next", env.keys.reward)]

    if is_goal:
        expected_reward = torch.full_like(reward, fill_value=env.default_goal_reward)
    elif is_dead_end:
        expected_reward = torch.full_like(reward, fill_value=custom_dead_end_reward)
    else:
        expected_reward = torch.full_like(reward, fill_value=env.default_reward)
    assert torch.allclose(reward, expected_reward, atol=0.01)
