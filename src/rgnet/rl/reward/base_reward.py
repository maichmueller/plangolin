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
