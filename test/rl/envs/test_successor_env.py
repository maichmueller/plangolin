from test.fixtures import *  # noqa: F401, F403
from typing import List

import mockito
import torch

import xmimir as xmi
from plangolin.rl.envs import SuccessorEnvironment
from plangolin.rl.reward import UnitReward
from plangolin.utils.misc import tolist


def create_successor_env(
    problem: xmi.XProblem, batch_size: int = 1
) -> SuccessorEnvironment:
    return SuccessorEnvironment(
        [xmi.XSuccessorGenerator(problem)],
        batch_size=torch.Size((batch_size,)),
        reward_function=UnitReward(deadend_reward=-1000),
    )


def test_successor_env_init(medium_blocks):
    problem: xmi.XProblem = medium_blocks[2]
    SuccessorEnvironment(
        [xmi.XSuccessorGenerator(problem)],
        batch_size=torch.Size((1,)),
        device=torch.device("cpu"),
        reward_function=UnitReward(deadend_reward=-1000),
        seed=42,
    )


def test_successor_env_initial_state(medium_blocks):
    problem = medium_blocks[2]
    env = create_successor_env(problem)
    td = env.reset()
    assert td.batch_size == (1,)
    states: List[xmi.XState] = tolist(td[env.keys.state])
    assert len(states) == 1
    assert not any(states[0].unsatisfied_literals(problem.initial_literals()))


def test_successor_env_is_goal(medium_blocks):
    space, _, problem = medium_blocks
    successor_gen = space.successor_generator
    goal_state: xmi.XState = next(space.goal_states_iter())
    env = SuccessorEnvironment(
        [successor_gen],
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=0.9),
    )
    td = env.reset(states=[goal_state])
    assert env.is_goal(td[env.keys.instance][0], goal_state)


def test_rollout(medium_blocks):
    env = create_successor_env(medium_blocks[2], batch_size=2)
    rollout = env.rollout(max_steps=4, break_when_any_done=False)
    assert rollout.batch_size == (2, 4)
    states = tolist(rollout[env.keys.state])
    assert len(states) == 2
    assert all(len(ls) == 4 for ls in states)


def test_dead_end(medium_blocks):
    problem: xmi.XProblem = medium_blocks[2]
    mocked_action_generator = mockito.mock(
        {"problem": problem, "generate_actions": lambda state: []},
        spec=xmi.XActionGenerator,
    )
    successor_gen = xmi.XSuccessorGenerator(problem)
    successor_gen.action_generator = mocked_action_generator

    env = SuccessorEnvironment(
        [successor_gen],
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(deadend_reward=-1000),
    )
    td = env.reset()
    transitions: List[List[xmi.XTransition]] = tolist(td[env.keys.transitions])
    initial_state = successor_gen.initial_state
    assert len(transitions) == 1 and all(len(ls) == 1 for ls in transitions)
    assert all(
        t.source == initial_state and t.target == initial_state and t.action is None
        for t in transitions[0]
    )
