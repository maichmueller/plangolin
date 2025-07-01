from abc import abstractmethod
from copy import copy
from typing import Sequence

import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer

import xmimir as xmi
from xmimir import XLiteral
from xmimir.wrappers import CustomProblem

from .successor_env import SuccessorEnvironment


class HindsightStrategy:
    """
    Base class for strategies that determine the hindsight goals.
    Subclasses should implement the `get_hindsight_goals` method.
    """

    @abstractmethod
    def __call__(self, state: xmi.XState) -> tuple[XLiteral]:
        """
        Returns a list of hindsight goals for the given state.
        :param state: The current state.
        :return: A list of hindsight goal literals.
        """
        ...


class RandomSubgoalHindsightStrategy(HindsightStrategy):
    """
    A simple strategy that randomly selects a goal from the state.
    This is a placeholder and should be replaced with a more sophisticated strategy.
    """

    def __init__(self, max_conjunction: int = 10, seed: int = None):
        """
        Initializes the random subgoal hindsight strategy.
        :param seed: Optional seed for reproducibility.
        """
        self.max_conjunction = max_conjunction
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

    def __call__(self, state: xmi.XState) -> tuple[XLiteral]:
        """
        Returns a random goal from the state.
        :param state: The current state.
        :return: A list containing a single random goal literal.
        """
        satisfied_goals = tuple(state.satisfied_literals(state.problem.goal()))
        if not satisfied_goals:
            return state.problem.goal()
        nr_goals = self.rng.integers(1, np.arange(len(satisfied_goals)))
        selected: np.ndarray = self.rng.choice(
            satisfied_goals, size=nr_goals, replace=False
        )
        return tuple(selected)


class HindsightEnvironment(SuccessorEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relabelling_succ_gens = [
            copy(instance) for instance in self._all_instances
        ]

    def relabel(
        self, tensordict: TensorDict, hindsight_goals: Sequence[tuple[XLiteral]]
    ) -> TensorDict:
        """
        Relabels the tensordict with hindsight goals.

        :param tensordict: The rollout tensordict to relabel.
            Shape: (T, B, ...), where T is the time dimension and B is the batch size.
        :param hindsight_goals: A sequence of tuples containing the hindsight goals for each instance in the batch.
            Shape: (B, G), where B is the batch size and G is the number of goals in the conjunction for this instance.

        :return: The relabeled tensordict.
        """
        td_bs = tensordict.batch_size  # a tuple, e.g. (T, B) or (T, B1, B2)
        env_bs = self.env.batch_size  # e.g. (B,) or (B1, B2)
        match len(td_bs) > len(env_bs):
            case x if x < 0:
                raise ValueError(
                    "The tensordict batch size is smaller than the environment batch size. "
                    "This is unexpected and likely an error in the tensordict or environment setup."
                )
            case x if x == 0:
                # If the tensordict batch size is equal to the environment batch size,
                assert all(a == b for a, b in zip(td_bs, env_bs))
                first_dim_is_time = False
            case x if x > 1:
                assert all(a == b for a, b in zip(td_bs[1:], env_bs, strict=True))
                first_dim_is_time = True
            case _:
                raise ValueError("This is unreachable code.")

        for succ_gen, hs_goal in zip(self._relabelling_succ_gens, hindsight_goals):
            succ_gen.problem = CustomProblem(
                succ_gen.problem,
                goal=hs_goal,
            )
        transitions = tensordict["action"]
        relabelling_instances = (
            [self._relabelling_succ_gens] * (td_bs[0] if first_dim_is_time else 1),
        )
        assert len(transitions) == len(relabelling_instances), (
            "The number of transitions does not match the number of relabelling instances. "
            f"Transitions: {len(transitions)}, Instances: {len(relabelling_instances)}"
        )
        rewards, done = self.get_reward_and_done_multi(
            transitions,
            instances=relabelling_instances,
        )
        out_td = tensordict.clone(recursive=False)  # shallow copy suffices
        out_td["reward"] = rewards
        out_td["done"] = done
        return out_td


class HERReplayBuffer(TensorDictReplayBuffer):
    def __init__(self, env, hindsight_strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hindsight_strategy = hindsight_strategy
        self.env = env

    def extend(
        self,
        tensordict: TensorDict,
    ) -> None:
        """
        Override to relabel once per-trajectory on insertion.
        """
        final_obs_list = tensordict[("next", "observation")][-1]
        new_goals = [self.hindsight_strategy(final_obs) for final_obs in final_obs_list]
        tensordict = self.env.relabel(tensordict, new_goals)
        super().extend(tensordict)
