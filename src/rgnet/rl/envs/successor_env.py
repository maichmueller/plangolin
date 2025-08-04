import warnings
from typing import Iterable, List, Tuple

from xmimir import XLiteral, XState, XSuccessorGenerator, XTransition

from .planning_env import PlanningEnvironment


class SuccessorEnvironment(PlanningEnvironment[XSuccessorGenerator]):
    def __init__(self, generators: Iterable[XSuccessorGenerator], *args, **kwargs):
        super().__init__(list(generators), *args, **kwargs)

    def transitions_for(
        self,
        active_instance: XSuccessorGenerator,
        state: XState,
    ) -> List[XTransition]:
        return list(active_instance.successors(state))

    def initial_for(
        self,
        active_instance: XSuccessorGenerator,
    ) -> Tuple[XState, List[XLiteral]]:
        return active_instance.initial_state, list(active_instance.problem.goal())

    def is_goal(self, active_instance: XSuccessorGenerator, state: XState) -> bool:
        if state.problem != active_instance.problem:
            warnings.warn("State does not belong to the active instance.")
        return state.is_goal(active_instance.problem.goal())
