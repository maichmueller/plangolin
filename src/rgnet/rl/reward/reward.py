import abc
from typing import Sequence

import torch

from xmimir import StateLabel, XAction, XTransition


class RewardFunction:
    """
    Base class for reward functions.

    Reward functions are used to calculate the reward of a state transition.
    A transition is defined by the current state, the action taken and the next state.
    They can optionally take in an info dictionary (e.g.dead-end info or goal info).
    A reward function has to be able to calculate the reward with only this information.
    """

    def __init__(self, *, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = device

    @abc.abstractmethod
    def __call__(
        self, transitions: Sequence[XTransition], labels: Sequence[StateLabel]
    ):
        """
        Abstract method for calculating the reward of a transition.

        :param transitions: A sequence of transitions.
        :param labels: A sequence of labels for the source state of each transition.
            Requires the same length as transitions.
        """
        ...

    @abc.abstractmethod
    def __eq__(self, other): ...


class UniformActionReward(RewardFunction):
    """
    A reward function that returns a reward (cost) of -1.0 for every primitive action taken.
    """

    def __init__(
        self,
        gamma: float | None = None,
        deadend_reward: float | None = None,
        goal_reward: float = 0.0,
        regular_reward: float = -1.0,
        *,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
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
            self.regular_reward == other.regular_reward
            and self.goal_reward == other.goal_reward
            and self.deadend_reward == other.deadend_reward
            and self.gamma == other.gamma
        )

    def __call__(
        self, transitions: Sequence[XTransition], labels: Sequence[StateLabel]
    ):
        rewards = []
        for transition, label in zip(transitions, labels):
            match label:
                case StateLabel.deadend:
                    rewards.append(self.deadend_reward)
                case StateLabel.goal:
                    rewards.append(self.goal_reward)
                case _:
                    if isinstance(transition.action, Sequence):
                        rewards.append(
                            self._reward_macro_action(transition.action, label)
                        )
                    else:
                        rewards.append(self.regular_reward)

        return torch.tensor(rewards, dtype=torch.float, device=self.device)

    def _reward_macro_action(self, macro: Sequence[XAction], label: StateLabel):
        return self.regular_reward


class FactoredMacroReward(UniformActionReward):
    """
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
        return super().__eq__(other) and self.factor == other.factor

    def _reward_macro_action(self, macro: Sequence[XAction], label: StateLabel):
        return self.regular_reward - len(macro) / self.factor


class DiscountedMacroReward(UniformActionReward):
    """
    A reward function that returns a reward of

    .. math::
        \sum_{i=1}^{n} \gamma^{i-1} = (\gamma^n - 1) / (1 - \gamma)

    for a macro action taken.
    """

    def __init__(
        self,
        *args,
        gamma_macros: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma_macros = gamma_macros if gamma_macros is not None else self.gamma

    def __eq__(self, other):
        return super().__eq__(other) and self.gamma_macros == other.gamma_macros

    def _reward_macro_action(self, macro: Sequence[XAction], label: StateLabel):
        match length := len(macro):
            case 0:
                raise ValueError("Cannot compute cost for empty action list")
            case 1:
                return -1.0
            case 2:
                return -(1.0 + self.gamma_macros)
            case _:
                return (1.0 - self.gamma**length) / (1.0 - self.gamma)
