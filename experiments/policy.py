import abc
import logging
from typing import Callable, List, Optional, Tuple

import pymimir as mi
import torch
from torch import Generator


class Policy(abc.ABC):

    @abc.abstractmethod
    def __call__(self, state: mi.State, applicable_actions: List[mi.Action]):
        pass

    def run(
        self, problem: mi.Problem, max_steps: int
    ) -> Optional[Tuple[List[mi.Action], int]]:

        plan = []
        steps = 0
        succ = mi.GroundedSuccessorGenerator(problem)
        state = problem.create_state(problem.initial)
        while not state.literals_hold(problem.goal) and steps <= max_steps:
            action, next_state = self(state, succ.get_applicable_actions(state))
            logging.debug(f"Selected {action} in state {state}")
            plan.append(action)
            state = next_state
            steps += 1
        if not state.literals_hold(problem.goal):
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
        self, state: mi.State, applicable_actions: List[mi.Action]
    ) -> Tuple[mi.Action, mi.State]:
        action_targets = [(a, a.apply(state)) for a in applicable_actions]
        idx = torch.randint(0, len(action_targets), (1,), generator=self.rng)[0]

        return action_targets[idx]


class ValuePolicy(Policy):

    def __init__(self, value_function: Callable[[mi.State], float]):
        self.value_function = value_function

    def __call__(
        self, state: mi.State, applicable_actions: List[mi.Action]
    ) -> Tuple[mi.Action, mi.State]:
        action_targets = [(a, a.apply(state)) for a in applicable_actions]
        return min(action_targets, key=lambda a_t: self.value_function(a_t[1]))

    def evaluate_actions(self, state: mi.State, succ: mi.SuccessorGenerator):
        """Return a list containing (Action, next-state, value(next-state) tuples.
        Sorted by the value (smallest to highest)
        """
        actions = succ.get_applicable_actions(state)
        action_targets: List[Tuple[mi.Action, mi.State]] = [
            (a, a.apply(state)) for a in actions
        ]
        a_t_v = [
            (action, next_state, self.value_function(next_state))
            for action, next_state in action_targets
        ]
        return sorted(a_t_v, key=lambda atv: atv[2])
