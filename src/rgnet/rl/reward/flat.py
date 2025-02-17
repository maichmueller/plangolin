import abc
from typing import Any, Sequence

import torch

from xmimir import XTransition


class RewardFunction(torch.nn.Module):
    """
    Base class for reward functions.

    Reward functions are used to calculate the reward of a state transition.
    A transition is defined by the current state, the action taken and the next state.
    They can optionally take in an info dictionary (e.g.dead-end info or goal info).
    A reward function has to be able to calculate the reward with only this information.
    """

    def __init__(self):
        self._device_dummy = torch.nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self._device_dummy.device

    @abc.abstractmethod
    def forward(
        self, transitions: Sequence[XTransition], info: dict[str, Any] | None = None
    ): ...


class PrimitiveActionReward(RewardFunction):
    """
    A reward function that returns a reward (cost) of -1.0 for every primitive action taken.
    """

    def __init__(
        self,
        goal_reward: float = 0.0,
        deadend_reward: float | None = None,
        gamma: float | None = None,
    ):
        super().__init__()
        self.goal_reward = goal_reward
        self.default_reward = -1.0
        if deadend_reward is None and gamma is None:
            raise ValueError("Either deadend_reward or gamma has to be set.")
        if deadend_reward is not None:
            self.deadend_reward = deadend_reward
        else:
            self.deadend_reward = 1.0 / (gamma - 1.0)

    def forward(
        self,
        transitions: Sequence[XTransition],
        info: Sequence[TransitionCategory] | None = None,
    ):
        out = torch.ones(len(transitions), dtype=torch.float, device="cpu")
        if info is None:
            return (out * -1.0).to(self.device)
        else:
            rewards = []
            dead_end_rewards = torch.where(
                condition=~is_dead_end,
                input=default_reward,
                other=self._dead_end_reward,
            )
            rewards = torch.where(
                condition=~is_goal,
                input=dead_end_rewards,
                other=self._goal_reward,
            )

            torch.where(
                torch.tensor(
                    [info["dead_end"] for _ in transitions], device=self.device
                ),
                self.deadend_reward,
                out * -1.0,
            )
        for transition in transitions:
            if transition.dead_end:
                rewards.append(self.deadend_reward)
            elif transition.goal:
                rewards.append(self.goal_reward)
            else:
                rewards.append(-1.0)
        return torch.tensor(rewards, dtype=torch.float)
