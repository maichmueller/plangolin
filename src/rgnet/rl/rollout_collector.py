from collections import OrderedDict
from math import ceil
from typing import Callable, Iterator, List, Optional

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.collectors import DataCollectorBase
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.batched_envs import BatchedEnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from rgnet.rl.envs import ExpandedStateSpaceEnv, MultiInstanceStateSpaceEnv
from rgnet.rl.envs.expanded_state_space_env import IteratingReset, WeightedRandomReset
from xmimir import XStateSpace


class RolloutCollector(DataCollectorBase):
    """
    A collector that generates rollouts from a policy and environment.
    The collector will return one rollout per iteration (next call).
    A complete environment reset will occur at the start of each batch.
    After collecting num_batches iterations the collector has to be reset.
    NOTE that currently, the first batch after reset was called can be different to the first batch before reset was called.
    This is due to the fact that InstanceReplacementStrategy and ResetStrategy are not reset alongside.
    :param environment: The environment to collect rollouts from.
    :param policy: The policy to use for generating actions.
        If None provided a random policy will be used
    :param rollout_length: The length of the rollout. Each batch entry will have this length in time.
        If a done-state was encountered before a reset for that batch entry will be performed.
    :param num_batches: Number of batches to generate. If -1 will generate indefinitely.
    :param exploration_type: The exploration type to use for the policy.
    """

    def __init__(
        self,
        environment: EnvBase,
        policy: (
            Callable[[TensorDictBase], TensorDictBase | None] | TensorDictModule | None
        ),
        rollout_length: int,
        num_batches: int = -1,
        exploration_type: ExplorationType = ExplorationType.RANDOM,
    ):
        self.policy = policy
        self.env = environment
        self.rollout_length = rollout_length
        self.num_batches = num_batches
        self.exploration_type = exploration_type
        self._iter = 0
        self.init_random_frames = 0  # required by Trainer

    def iterator(self) -> Iterator[TensorDictBase]:
        return iter(self)

    def __next__(self):
        if self.num_batches == -1 or self._iter < self.num_batches:
            with set_exploration_type(self.exploration_type):
                self._iter += 1
                return self.env.rollout(
                    max_steps=self.rollout_length,
                    policy=self.policy,
                )
        else:
            raise StopIteration

    def __iter__(self) -> Iterator[TensorDictBase]:
        self.reset()
        return self

    def reset(self) -> None:
        self._iter = 0

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return self.env.set_seed(seed, static_seed)

    def state_dict(self) -> OrderedDict:
        if isinstance(self.env, TransformedEnv):
            env_state_dict = self.env.transform.state_dict()
        elif isinstance(self.env, BatchedEnvBase):
            env_state_dict = self.env.state_dict()
        else:
            env_state_dict = OrderedDict()

        if self.policy is not None and hasattr(self.policy, "state_dict"):
            policy_state_dict = self.policy.state_dict()
            state_dict = OrderedDict(
                policy_state_dict=policy_state_dict,
                env_state_dict=env_state_dict,
            )
        else:
            state_dict = OrderedDict(env_state_dict=env_state_dict)

        state_dict.update({"iter": self._iter})

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        """Loads a state_dict on the environment and policy.

        Args:
            state_dict (OrderedDict): ordered dictionary containing the fields
                `"policy_state_dict"` and :obj:`"env_state_dict"`.

        """
        strict = kwargs.get("strict", True)
        if strict or "env_state_dict" in state_dict:
            self.env.load_state_dict(state_dict["env_state_dict"], **kwargs)
        if strict or "policy_state_dict" in state_dict:
            self.policy.load_state_dict(state_dict["policy_state_dict"], **kwargs)
        self._iter = state_dict["iter"]

    def shutdown(self):
        pass


def build_from_spaces(
    spaces: XStateSpace | List[XStateSpace],
    batch_size: int,
    rollout_length: int = 1,
    num_batches: int | None = None,
    env_kwargs: Optional[dict] = None,
    policy: Callable[[TensorDictBase], TensorDictBase | None] | TensorDictModule = None,
    exploration_type: ExplorationType = ExplorationType.RANDOM,
):
    """
    Build a rollout collector from a list of state spaces.
    The collector will be setup to go over each state of each instance once (in expectation).
    Note that this only holds if rollout_length = 1 is passed and num_batches is None.
    The batch size of the environment is the batch size of each collected rollout.
    """
    if env_kwargs is None:
        env_kwargs = {}
    spaces = [spaces] if isinstance(spaces, XStateSpace) else spaces

    env: MultiInstanceStateSpaceEnv
    # test if all spaces in the list are the same object
    if all([space is spaces[0] for space in spaces]):
        env = ExpandedStateSpaceEnv(
            spaces[0],
            batch_size=torch.Size((batch_size,)),
            reset_strategy=IteratingReset(),
            **env_kwargs,
        )
    else:
        env = MultiInstanceStateSpaceEnv(
            spaces,
            batch_size=torch.Size((batch_size,)),
            reset_strategy=IteratingReset(),
            **env_kwargs,
        )
    env.make_replacement_strategy(WeightedRandomReset)

    if num_batches is None:
        num_batches = ceil(sum(len(space) for space in spaces) / float(batch_size))

    return RolloutCollector(
        environment=env,
        policy=policy,
        num_batches=num_batches,
        rollout_length=rollout_length,
        exploration_type=exploration_type,
    )
