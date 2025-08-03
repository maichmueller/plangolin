from abc import abstractmethod
from copy import copy
from typing import Sequence

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import TensorDictReplayBuffer

import xmimir as xmi
from rgnet.utils.batching import expand_sequence
from rgnet.utils.data import map_dim, nested_to_array
from xmimir import XLiteral
from xmimir.wrappers import CustomProblem, XAtom

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

    def __init__(
        self,
        max_conjunction: int | None = None,
        random_goal_chance: float = 0.05,
        seed: int = None,
    ):
        """
        Initializes the random subgoal hindsight strategy.
        :param seed: Optional seed for reproducibility.
        :param max_conjunction: Maximum number of goals to select from the state.
            If None, it will select all goals.
        :param random_goal_chance: Minimum probability of selecting a random fluent or derived atom from the true atoms in
            the current state as goal. May be effectively higher if the goal rarely has satisfied literals in the state.
        """
        self.max_conjunction = max_conjunction or float("inf")
        self.random_goal_chance = random_goal_chance
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

    def __call__(self, state: xmi.XState) -> tuple[XLiteral]:
        """
        Returns a random subset of goals from the state.
        :param state: The current state.
        :return: A list containing a random subset of goal literals.
        """
        satisfied_goals = tuple(state.satisfied_literals(state.problem.goal()))
        if self.rng.random() < self.random_goal_chance:
            # Select a random fluent or derived atom from the state
            atoms = tuple(state.atoms(with_statics=False))
            random_goal: XAtom = atoms[
                int(self.rng.choice(np.arange(0, len(atoms)), size=1, replace=False))
            ]
            return (XLiteral.make_hollow(atom=random_goal, negated=False),)
        else:
            if not satisfied_goals:
                return state.problem.goal()
            nr_goals = self.rng.integers(
                1,
                max(min(len(satisfied_goals), self.max_conjunction), 1),
                endpoint=True,
            )
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
        assert tensordict.names[-1] == "time", (
            "The tensordict must have a 'time' dimension as the last dimension. "
            "This is required for the relabelling process to work correctly."
        )
        *td_bs, td_ts = tensordict.batch_size  # a sequence, (B, T) or (B1, B2, ..., T)
        env_bs = self.batch_size  # (B,) or (B1, B2, ...)

        for succ_gen, hs_goal in zip(self._relabelling_succ_gens, hindsight_goals):
            succ_gen.problem = CustomProblem(
                succ_gen.problem,
                goal=hs_goal,
            )
        transitions = tensordict[self.keys.action]
        relabelling_instances = expand_sequence(
            env_bs, td_ts, self._relabelling_succ_gens
        )
        rewards, done = self.get_reward_and_done_multi(
            transitions,
            instances=relabelling_instances,
        )
        # are the nested tensors actually dense?
        rewards_nested_shape_diffs = torch.diff(rewards.offsets())
        actually_dense: bool = (
            (rewards_nested_shape_diffs == rewards_nested_shape_diffs[0]).all().item()
        )

        # self._restore_achieved_goals(tensordict, expanded_hindsight_goals, done)
        corrected_hindsight_goals, corrected_rewards, corrected_done = (
            self._account_for_achieved_hs_goals(
                tensordict, hindsight_goals, rewards, done, dense=actually_dense
            )
        )
        out_td = tensordict.clone(recurse=False)  # shallow copy suffices
        out_td[self.keys.goals] = corrected_hindsight_goals
        out_td[self.keys.done] = corrected_done
        # overwrite both entries for done so that no confusion arises by accident
        out_td["next", self.keys.done] = corrected_done
        out_td["next", self.keys.reward] = corrected_rewards
        return out_td

    def _account_for_achieved_hs_goals(
        self,
        tensordict: TensorDict,
        hindsight_goals: Sequence,
        rewards: torch.nested.Tensor,
        done: torch.nested.Tensor,
        dense: bool = True,
    ):
        """
        Restores the original goals for the remainder of each trajectory after its first done is noted.
        This is necessary because the hindsight goals may be achieved after some steps in the trajectory,
        thus prompting a reset in a normal circumstance. Since we did not reset the environment for this
        goal during rollout, we need to restore the original goals for the remainder of the trajectory to
        have a meaningful training signal in those segments.

        :param tensordict: The tensordict containing the trajectory data.
        :param hindsight_goals: The hindsight goals to be relabelled. Expected to be a nested sequence of goal-literal tuples.
        :param rewards: The rewards tensor, expected to be a torch.nested Tensor.
        :param done: The done tensor, expected to be a torch.nested Tensor.
        :param dense: Whether the rewards and done tensors are actually dense.
            If False, an error will be raised at the moment. The non-dense case is not yet implemented.

        :return: A tuple containing the corrected hindsight goals, rewards, and done tensors.
        :raises ValueError: If the rewards tensor is not dense.
        """
        if not dense:
            raise ValueError(
                "The rewards tensor is not dense. This is currently not handled by the relabelling process."
            )
            # return cascade(
            #     fn=torch.stack,
            #     nested=map_last_dim(NonTensorData, hindsight_goals.tolist()),
            #     exclude_dims=[-1],
            # )
        else:
            dense_done = done.to_padded_tensor(padding=False)
            dense_rewards = rewards.to_padded_tensor(padding=torch.nan)
            # Store originals once (before mutating)
            original_goals = tensordict["next", self.keys.goals]
            original_rewards = tensordict["next", self.keys.reward]
            original_done = tensordict[self.keys.done]
            batch_size = tensordict.batch_size
            *data_batch_size, time_size = batch_size

            hindsight_goals_array = nested_to_array(
                data_batch_size, hindsight_goals, dtype=np.dtype("object")
            )
            original_goals_array = nested_to_array(
                batch_size, original_goals, dtype=np.dtype("object")
            )
            # broadcast to shape including time dimension (make copy)
            hindsight_goals_array = np.repeat(
                hindsight_goals_array[..., None], time_size, axis=-1
            )
            # find all entries where done is True, shape: (#true entries, tensordict.ndim)
            where_done = dense_done.nonzero(as_tuple=False)
            # trajectory-parts after a `done` signal are no longer valid for the hindsight goal, since we did not reset
            # the environment for this goal. Hence, we restore the original goals for these parts.
            prefix = where_done[:, :-1]  #  prefix indices
            t_idx = where_done[:, -1]  # time indices
            # compute flat index for (b1,b2,...,bn)
            prefix_idx = torch.from_numpy(
                np.ravel_multi_index(prefix.T.numpy(), dims=data_batch_size)
            )  # (k,)

            # initialize result with "no true" sentinel; here using t (out of range)
            first_t_flat = torch.full(
                data_batch_size, time_size, device=done.device, dtype=torch.long
            )
            # scatter-reduce the time indices (take min t per prefix)
            first_t_flat.scatter_reduce_(
                0, prefix_idx, t_idx, reduce="amin", include_self=True
            )
            # reshape back to multi-dim prefix shape
            first_t = first_t_flat.view(
                *data_batch_size
            )  # now has earliest done time per prefix
            first_t = first_t.masked_fill(first_t == time_size, -1)

            # Build t_range shaped to broadcast to (*batch_prefix, time)
            t_range = torch.arange(time_size, device=done.device)  # (time,)
            # reshape first_t to (*batch_prefix, 1)
            first_t_broadcasted = torch.broadcast_to(first_t.unsqueeze(-1), batch_size)

            # broadcast t_range to (*batch_prefix, time)
            # Compare: for each prefix, mark t >= first_t (and exclude prefixes with no true)
            mask_ge = torch.logical_or(
                t_range >= first_t_broadcasted + 1,
                first_t_broadcasted
                == 0,  # if goal is true at start of trajectory, replace
            )  # broadcasts to (..., time)
            valid_prefix = (
                first_t_broadcasted >= 0
            )  # (..., 1), will broadcast over time

            mask = mask_ge & valid_prefix  # (..., time)
            # apply assignment
            dense_done[mask] = original_done[mask].squeeze()
            dense_rewards[mask] = original_rewards[mask].squeeze()
            hindsight_goals_array[mask] = original_goals_array[mask]
            return hindsight_goals_array, dense_rewards, dense_done


class HERReplayBuffer(TensorDictReplayBuffer):
    def __init__(self, env: HindsightEnvironment, hindsight_strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hindsight_strategy = hindsight_strategy
        self.env = env

    def add(self, data: TensorDictBase) -> int:
        raise NotImplementedError(
            f"{self.__class__.__qualname__} does not support adding single steps. "
            "Use `extend` to add entire trajectories instead. When using a collector, "
            "set `extend_buffer=True` to automatically extend the buffer with trajectories and avoid this error."
        )

    def extend(
        self,
        tensordict: TensorDict,
    ) -> None:
        """
        Override to relabel once per-trajectory on insertion.
        """
        final_obs_list = tensordict[("next", self.env.keys.state)]
        has_time_dim = "time" in tensordict.names
        new_goals = map_dim(
            lambda final_obs: self.hindsight_strategy(final_obs[-1]),
            final_obs_list,
            dim=-1 - int(has_time_dim),
        )
        tensordict = self.env.relabel(tensordict, new_goals)
        super().extend(tensordict)
