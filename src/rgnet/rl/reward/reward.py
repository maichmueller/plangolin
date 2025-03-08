import abc
from typing import Any, Sequence

import torch

from xmimir import StateLabel, XTransition


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
        else:
            self.deadend_reward = 1.0 / (gamma - 1.0)

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
                    rewards.append(self.regular_reward)
        return torch.tensor(rewards, dtype=torch.float, device=self.device)


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
                            self.regular_reward - len(transition.action) / self.factor
                        )
                    else:
                        rewards.append(self.regular_reward)
        return torch.tensor(rewards, dtype=torch.float, device=self.device)
