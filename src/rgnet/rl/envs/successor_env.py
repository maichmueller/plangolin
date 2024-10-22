from typing import List, Optional, Tuple

import pymimir as mi
import torch

from rgnet.rl.envs.manual_transition import MTransition
from rgnet.rl.envs.planning_env import PlanningEnvironment


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
        self, active_instance: Tuple[mi.SuccessorGenerator, mi.Problem], state: mi.State
    ) -> List[mi.Transition]:
        generator, _ = active_instance

        actions = generator.get_applicable_actions(state)
        return [MTransition(state, action, action.apply(state)) for action in actions]

    def initial_for(
        self, active_instance: Tuple[mi.SuccessorGenerator, mi.Problem]
    ) -> Tuple[mi.State, List[mi.Literal]]:
        _, problem = active_instance
        return problem.create_state(problem.initial), problem.goal

    def is_goal(self, active_instance: Tuple, state: mi.State) -> bool:
        return state.literals_hold(active_instance[1].goal)
