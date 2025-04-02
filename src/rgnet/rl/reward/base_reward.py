import abc
from typing import List, Sequence

from xmimir import StateLabel, XTransition


class RewardFunction:
    """
    Base class for reward functions.

    Reward functions are used to calculate the reward of a state transition.
    A transition is defined by the current state, the action taken and the next state.
    They can optionally take in an info dictionary (e.g.dead-end info or goal info).
    A reward function has to be able to calculate the reward with only this information.
    """

    def __init__(
        self,
        gamma: float | None = None,
        deadend_reward: float | None = None,
    ):
        """
        Initialize the reward function.
        """
        if deadend_reward is None and gamma is None:
            raise ValueError("Either deadend_reward or gamma has to be set.")
        if deadend_reward is not None:
            self.deadend_reward = deadend_reward
            self.gamma = 1.0 / deadend_reward + 1.0
        else:
            self.deadend_reward = 1.0 / (gamma - 1.0)
            self.gamma = gamma

    @abc.abstractmethod
    def __call__(
        self, transitions: Sequence[XTransition], labels: Sequence[StateLabel]
    ) -> List[float]:
        """
        Abstract method for calculating the reward of a transition.

        :param transitions: A sequence of transitions.
        :param labels: A sequence of labels for the source state of each transition.
            Requires the same length as transitions.
        """
        ...

    @abc.abstractmethod
    def __eq__(self, other): ...

    def __hash__(self):
        return hash(
            f"{self.__class__.__name__}({self.gamma:.9f}, {self.deadend_reward:.9f})"
        )
