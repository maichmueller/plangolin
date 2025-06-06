import warnings
from typing import Iterable, List, Tuple

import xmimir as xmi
from rgnet.rl.envs.planning_env import PlanningEnvironment
from xmimir import XLiteral, XState


class SuccessorEnvironment(PlanningEnvironment[xmi.XSuccessorGenerator]):
    def __init__(self, generators: Iterable[xmi.XSuccessorGenerator], *args, **kwargs):
        super().__init__(list(generators), *args, **kwargs)

    def transitions_for(
        self,
        active_instance: xmi.XSuccessorGenerator,
        state: xmi.XState,
    ) -> List[xmi.XTransition]:
        return list(active_instance.successors(state))

    def initial_for(
        self,
        active_instance: xmi.XSuccessorGenerator,
    ) -> Tuple[XState, List[XLiteral]]:
        return active_instance.initial_state, list(active_instance.problem.goal())

    def is_goal(
        self, active_instance: xmi.XSuccessorGenerator, state: xmi.XState
    ) -> bool:
        if state.problem != active_instance.problem:
            raise warnings.warn("State does not belong to the active instance.")
        return state.is_goal(active_instance.problem.goal())
