from collections import OrderedDict
from typing import Callable, Iterator

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.collectors import DataCollectorBase
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.batched_envs import BatchedEnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type


class RolloutCollector(DataCollectorBase):

    def __init__(
        self,
        environment: EnvBase,
        policy: Callable[[TensorDictBase], TensorDictBase | None] | TensorDictModule,
        rollout_length: int,
        exploration_type: ExplorationType = ExplorationType.RANDOM,
        custom_reset_func: Callable[[EnvBase], TensorDict] | None = None,
    ):
        self.policy = policy
        self.env = environment
        self.rollout_length = rollout_length
        self.custom_reset_func = custom_reset_func
        self.exploration_type = exploration_type
        self._iter = 0
        self.init_random_frames = 0  # required by Trainer

    def iterator(self) -> Iterator[TensorDictBase]:
        with set_exploration_type(self.exploration_type):
            while True:
                if self.custom_reset_func is not None:
                    td = self.custom_reset_func(self.env)
                else:
                    td = self.env.reset()
                self._iter += 1
                yield self.env.rollout(
                    max_steps=self.rollout_length,
                    policy=self.policy,
                    auto_reset=False,
                    tensordict=td,
                )

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return self.env.set_seed(seed, static_seed)

    def state_dict(self) -> OrderedDict:
        if isinstance(self.env, TransformedEnv):
            env_state_dict = self.env.transform.state_dict()
        elif isinstance(self.env, BatchedEnvBase):
            env_state_dict = self.env.state_dict()
        else:
            env_state_dict = OrderedDict()

        if hasattr(self.policy, "state_dict"):
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
