import math
from typing import Sequence

from xmimir import StateLabel, XAction, XTransition

from .base_reward import RewardFunction


def _n_step_discounted_reward(steps: int, gamma: float, regular_reward: float):
    r"""
    Computes the discounted reward for a sequence of actions (macro action).

    Each action in the macro is discounted by a factor of :math:`\gamma` compared to the previous action and
    rewarded with `regular_reward`. The reward for the macro is then the sum of the discounted rewards for each action.
    The concise formula for the discounted reward of a sequence of actions is:

    ..math::
        \sum_{i=1}^{n} r \cdot \gamma^{i-1} = r \cdot \frac{1 - \gamma^n}{1 - \gamma}
    """
    match steps:
        case 0:
            raise ValueError("Cannot compute cost for 0 steps.")
        case 1:
            return regular_reward
        case 2:
            return regular_reward + gamma * regular_reward
        case _:
            return regular_reward * (1.0 - gamma**steps) / (1.0 - gamma)


class UnitReward(RewardFunction):
    r"""
    A reward function that returns a reward (cost) of -1.0 for every primitive action taken.

    The reward for macros is abstracted to primitive reward as well. This can be changed in child classes by
    overriding the _reward_macro method.
    """

    def __init__(
        self,
        gamma: float | None = None,
        deadend_reward: float | None = None,
        goal_reward: float = 0.0,
        regular_reward: float = -1.0,
    ):
        self.regular_reward = regular_reward
        self.goal_reward = goal_reward
        if deadend_reward is None and gamma is None:
            raise ValueError("Either deadend_reward or gamma has to be set.")
        if deadend_reward is not None:
            self.deadend_reward = deadend_reward
            self.gamma = 1.0 / deadend_reward + 1.0
        else:
            self.deadend_reward = 1.0 / (gamma - 1.0)
            self.gamma = gamma

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            math.isclose(self.regular_reward, other.regular_reward, rel_tol=1e-9)
            and math.isclose(self.goal_reward, other.goal_reward, rel_tol=1e-9)
            and math.isclose(self.deadend_reward, other.deadend_reward, rel_tol=1e-9)
            and math.isclose(self.gamma, other.gamma, rel_tol=1e-9)
        )

    def __call__(
        self, transitions: Sequence[XTransition], labels: Sequence[StateLabel]
    ) -> list[float]:
        rewards = []
        for transition, label in zip(transitions, labels):
            match label:
                case StateLabel.deadend:
                    rewards.append(self.deadend_reward)
                case StateLabel.goal:
                    rewards.append(self.goal_reward)
                case _:
                    if isinstance(transition.action, Sequence):
                        rewards.append(self._reward_macro(transition, label))
                    else:
                        rewards.append(self.regular_reward)
        return rewards

    def _reward_macro(self, transition: XTransition, label: StateLabel):
        return _n_step_discounted_reward(
            len(transition.action), self.gamma, self.regular_reward
        )


class FlatReward(UnitReward):
    r"""
    A reward function that returns the primitive action reward for every action taken, even entire macros.
    """

    def _reward_macro(self, macro: Sequence[XAction], label: StateLabel):
        return self.regular_reward


class FactoredMacroReward(UnitReward):
    r"""
    A reward function that returns a reward of -(1 + len(actions) / factor) for a (macro) action taken.
    """

    def __init__(
        self,
        factor: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factor = factor

    def __eq__(self, other):
        return super().__eq__(other) and math.isclose(self.factor, other.factor)

    def _reward_macro(self, transition: XTransition, label: StateLabel):
        if len(transition.action) == 0:
            raise ValueError("Cannot compute reward for 0-length macro.")
        return self.regular_reward - (len(transition.action) - 1) / self.factor


class DiscountedMacroReward(UnitReward):
    r"""
    A reward function that returns a discounted reward for a macro action taken.

    The difference to `FactoredMacroReward` is that the factor is a fixed value based on the chosen gamma value, i.e.
    the infinite plan length for that gamma, and thus is parameter-free.

    We want to discount fractionally all steps after the first step of the macro (the first step is seen as a
    regular action).
    Our formula for the discounted reward of a sequence (a_0, a_1, ..., a_n) of actions is:

    .. math:
        r + \frac{\sum_{i=1}^{n} r \cdot \gamma^i}{\sum_{i=1}^{\infinity} r \cdot \gamma^i} = r \cdot (2 - \gamma^n)
    """

    def _reward_macro(self, transition: XTransition, label: StateLabel):
        n = len(transition.action) - 1
        if n == 0:
            raise ValueError("Cannot compute reward for 0-length macro.")
        return self.regular_reward * (2 - self.gamma**n)
