from typing import List, Optional, Tuple

import pymimir as mi
import torch

from rgnet.rl.envs.planning_env import InstanceType, PlanningEnvironment


class SuccessorEnvironment(
    PlanningEnvironment[Tuple[mi.SuccessorGenerator, mi.Problem]]
):

    def __init__(
        self,
        generators: List[mi.SuccessorGenerator],
        problems: List[mi.Problem],
        batch_size: torch.Size,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(list(zip(generators, problems)), batch_size, seed, device)

    def transitions_for(
        self, active_instance: int, state: mi.State
    ) -> List[mi.Transition]:
        generator, _ = self._active_instances[active_instance]

        actions = generator.get_applicable_actions(state)
        return [MTransition(state, action, action.apply(state)) for action in actions]

    def initial_for(
        self, active_instances: Tuple[mi.SuccessorGenerator, mi.Problem]
    ) -> Tuple[mi.State, List[mi.Literal]]:
        problem = active_instances[1]
        return problem.create_state(problem.initial), problem.goal

    def is_goal(self, active_instance: Tuple, state: mi.State) -> bool:
        return state.matches_all(active_instance[1].goal)


class MTransition(mi.Transition):
    """There is sadly no constructor for mi.Transition, which is really just a data class"""

    def __init__(self, source: mi.State, action: mi.Action, target: mi.State):
        super().__init__()
        self._source = source
        self._action = action
        self._target = target

    @property
    def action(self):
        return self._action

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __eq__(self, __value):
        return isinstance(__value, mi.Transition) and (
            __value.action == self.action
            and __value.source == self.source
            and __value.target == self.target
        )

    def __hash__(self):
        return hash((self.source, self.action, self.target))
