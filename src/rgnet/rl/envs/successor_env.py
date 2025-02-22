from typing import Iterable, List, Optional, Tuple

import torch

import xmimir as xmi
from rgnet.rl.envs.planning_env import PlanningEnvironment
from xmimir import XLiteral, XState


class SuccessorEnvironment(
    PlanningEnvironment[
        Tuple[xmi.StateRepository, xmi.GroundedApplicableActionGenerator, xmi.XProblem]
    ]
):

    def __init__(
        self,
        generators: Iterable[
            Tuple[xmi.StateRepository, xmi.GroundedApplicableActionGenerator]
        ],
        problems: Iterable[xmi.XProblem],
        batch_size: torch.Size,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(list(zip(generators, problems)), batch_size, seed, device)

    def transitions_for(
        self,
        active_instance: Tuple[
            xmi.StateRepository, xmi.GroundedApplicableActionGenerator, xmi.XProblem
        ],
        state: xmi.XState,
    ) -> List[xmi.XTransition]:
        state_repo, generator, _ = active_instance
        return [
            xmi.XTransition(
                state,
                xmi.XState(
                    state_repo.get_or_create_successor_state(state.base, action)
                ),
                action,
            )
            for action in generator.generate_applicable_actions(state.base)
        ]

    def initial_for(
        self,
        active_instance: Tuple[
            xmi.StateRepository, xmi.GroundedApplicableActionGenerator, xmi.XProblem
        ],
    ) -> Tuple[XState, List[XLiteral]]:
        state_repo, generator, problem = active_instance
        return state_repo.get_or_create_initial_state(), list(problem.goal())

    def is_goal(self, active_instance: Tuple, state: xmi.XState) -> bool:
        return not any(state.unsatisfied_literals(active_instance[2].goals()))
