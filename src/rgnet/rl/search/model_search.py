import itertools
from typing import Optional

import torch
from tensordict import TensorDictBase
from tensordict.nn import InteractionType, TensorDictModule
from torchrl.envs.utils import set_exploration_type

from rgnet.rl.embedding import NonTensorTransformedEnv
from rgnet.rl.envs import PlanningEnvironment, SuccessorEnvironment
from rgnet.rl.reward import RewardFunction
from rgnet.rl.search.env_transforms import (
    CycleAvoidingTransform,
    NoTransitionTruncationTransform,
)
from rgnet.rl.thundeRL.policy_gradient.cli import TestSetup
from rgnet.utils.plan import Plan, ProbabilisticPlan, analyze_cycles, compute_return
from xmimir import XProblem, XState, XSuccessorGenerator, iw


class ModelSearch:
    """
    Base class for a model-based search.

    Model-based refers to the search relying on a torch.nn.Module as a value-heuristic or state discriminator and not
    to a model of the environment.
    """

    env_keys = PlanningEnvironment.default_keys

    def __init__(
        self,
        test_setup: TestSetup,
        reward_function: RewardFunction,
        device: torch.device = torch.device("cpu"),
    ):
        self.test_setup = test_setup
        self.device: torch.device = device
        self.reward_function: RewardFunction = reward_function

    def successor_env(
        self,
        problem: XProblem,
    ) -> SuccessorEnvironment:
        """
        Create a successor environment for the given problem.
        If `iw_search` is provided, it will be used to create an IWSuccessorGenerator.
        Otherwise, a standard XSuccessorGenerator will be used.
        """
        if (iw_search := self.test_setup.iw_search) is not None:
            generator = iw.IWSuccessorGenerator(iw_search, problem)
        else:
            generator = XSuccessorGenerator(problem)
        return SuccessorEnvironment(
            generators=[generator],
            reward_function=self.reward_function,
            batch_size=torch.Size((1,)),
        )

    def rollout_on_env(
        self,
        env: PlanningEnvironment,
        actor: TensorDictModule,
        policy_mode: InteractionType | None = None,
        initial_state: XState | None = None,
        max_steps: int | None = None,
    ):
        env, initial = self.init_env(env, initial_state)
        actor = actor.to(self.device)
        if policy_mode is None:
            policy_mode = self.test_setup.exploration_type
        with set_exploration_type(policy_mode), torch.no_grad():
            return env.rollout(
                max_steps=max_steps or self.test_setup.max_steps,
                policy=actor,
                tensordict=initial,
            )

    def init_env(
        self,
        env: PlanningEnvironment,
        initial_state: XState | None = None,
    ) -> tuple[PlanningEnvironment, Optional[TensorDictBase]]:
        cycle_transform = CycleAvoidingTransform(self.env_keys.transitions)
        if self.test_setup.avoid_cycles:
            env = NonTensorTransformedEnv(
                env=env,
                transform=cycle_transform,
            )
            env = NonTensorTransformedEnv(
                env=env,
                transform=NoTransitionTruncationTransform(
                    self.env_keys.transitions,
                    self.env_keys.done,
                    self.env_keys.truncated,
                ),
            )
        initial_td = (
            env.reset(states=[initial_state] * env.batch_size[0])
            if initial_state
            else None
        )
        return env, initial_td

    def rollout_on_problem(self, problem: XProblem, **kwargs):
        return self.rollout_on_env(
            self.successor_env(problem),
            **kwargs,
        )

    def analyze(
        self,
        problem: XProblem,
        rollout: TensorDictBase,
        optimal_plan: Optional[Plan] = None,
    ) -> ProbabilisticPlan:
        problem: XProblem
        # Assert we only have one batch entry and the time dimension is the last
        assert rollout.batch_size[0] == 1
        assert rollout.names[-1] == "time"
        transitions = list(
            itertools.takewhile(
                lambda t: not t.source.is_goal(),
                rollout["action"][0],
            )
        )
        rl_return, cost = compute_return(self.reward_function.gamma, transitions)
        cycles = analyze_cycles(transitions)
        plan_result = ProbabilisticPlan(
            problem=problem,
            solved=rollout[("next", "terminated")].any().item(),
            average_probability=1.0,
            min_probability=1.0,
            transitions=transitions,
            rl_return=round(rl_return, 3),
            cost=cost,
            subgoals=len(transitions),
            cycles=cycles,
        )
        if optimal_plan is not None:
            rl_return_optimal, cost_optimal = compute_return(
                self.reward_function.gamma, optimal_plan.transitions
            )
            plan_result.optimal_transitions = optimal_plan.transitions
            plan_result.optimal_cost = round(cost_optimal, 4)
            plan_result.diff_cost_to_optimal = plan_result.cost - cost_optimal
            plan_result.diff_return_to_optimal = rl_return - rl_return_optimal
            for i, (plan_step, optimal_plan_step) in enumerate(
                zip(plan_result.transitions, optimal_plan.transitions)
            ):
                if not plan_step.target.semantic_eq(optimal_plan_step.target):
                    plan_result.deviation_from_optimal = i
                    break
        return plan_result
