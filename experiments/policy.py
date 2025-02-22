import abc
import logging
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import pymimir as mi
import torch
from torch import Generator

import xmimir as xmi


class Policy(abc.ABC):

    @abc.abstractmethod
    def __call__(self, state: xmi.XState, applicable_actions: Sequence[xmi.XAction]):
        pass

    def run(
        self, problem: xmi.XProblem, max_steps: int
    ) -> Optional[Tuple[List[xmi.XAction], int]]:

        plan = []
        steps = 0
        action_gen = xmi.XActionGenerator(problem)
        succ = xmi.XSuccessorGenerator(action_gen.grounder)
        state = succ.initial_state
        while not state.is_goal() and steps <= max_steps:
            action, next_state = self(state, succ.successors(state, action_gen))
            logging.debug(f"Selected {action} in state {state}")
            plan.append(action)
            state = next_state
            steps += 1
        if not state.is_goal():
            logging.info(f"Could not find a plan in {max_steps}.")
            return None
        plan_cost = sum(a.cost for a in plan)
        return plan, plan_cost


class RandomPolicy(Policy):

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = Generator()
        self.rng.manual_seed(seed)

    def __call__(
        self,
        state: mi.State,
        actions_and_successors: Iterable[tuple[xmi.XAction], xmi.XState],
    ) -> Tuple[xmi.XAction, xmi.XState]:
        action_targets = [(a, s) for a, s in actions_and_successors]
        idx = torch.randint(0, len(action_targets), (1,), generator=self.rng)[0]

        return action_targets[idx]


class ValuePolicy(Policy):

    def __init__(self, value_function: Callable[[xmi.XState], float]):
        self.value_function = value_function

    def __call__(
        self,
        state: xmi.XState,
        actions_and_successors: Iterable[tuple[xmi.XAction], xmi.XState],
    ) -> Tuple[xmi.XAction, xmi.XState]:
        action_targets = [(a, s) for a, s in actions_and_successors]
        return min(action_targets, key=lambda a_t: self.value_function(a_t[1]))

    def evaluate_actions(
        self,
        state: xmi.XState,
        action_gen: xmi.XActionGenerator,
        succ: xmi.XSuccessorGenerator,
    ):
        """Return a list containing (Action, next-state, value(next-state) tuples.
        Sorted by the value (smallest to highest)
        """
        action_targets: List[Tuple[xmi.XAction, xmi.XState]] = list(
            succ.successors(state, action_gen)
        )
        a_t_v = [
            (action, next_state, self.value_function(next_state))
            for action, next_state in action_targets
        ]
        return sorted(a_t_v, key=lambda atv: atv[2])
