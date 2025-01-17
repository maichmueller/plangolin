import abc
from __future__ import annotations

import dataclasses
from typing import List
import itertools
import warnings
from typing import List, Sequence, Tuple

import pymimir as mi
import torch
from tensordict import NestedKey, NonTensorStack, TensorDict
from torch import Tensor
from torchrl.modules import ValueOperator
from torchrl.objectives.value import TD0Estimator

from rgnet.rl import ActorCritic
from rgnet.rl.envs.planning_env import InstanceType, PlanningEnvironment
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack
from rgnet.utils.object_embeddings import ObjectEmbedding


def _get_time_step_of(batched_data: Sequence, time_step: int):
    return [timed[time_step] for timed in batched_data]


class RewardDoneProvider:

    @abc.abstractmethod
    def __call__(
        self, tensordict: TensorDict
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Return the reward and done entry for each possible transition over batch and time dimension"""
        pass

    @property
    @abc.abstractmethod
    def in_keys(self) -> List[NestedKey]:
        pass


class EnvironmentBasedRewardProvider(RewardDoneProvider):
    """
    Go over each state over time and batch dimension and add the reward and done for
    all possible transitions and not only the one taken.
    Note that the returned tensors will be on the same device as the environment.
    """

    def __init__(self, env: PlanningEnvironment):
        self._env = env

    @property
    def in_keys(self) -> List[NestedKey]:
        return [
            self._env.keys.state,
            self._env.keys.transitions,
            self._env.keys.instance,
        ]

    def __call__(self, tensordict: TensorDict):
        """
        :param tensordict: should contain:
            transitions: The outgoing transitions of type List[List[List[pymimir.Transition]]].
            Dimensions are batch_size, time-steps, number of successors.
            current_states: The batched current states.
            instances: The instance from which the states and transitions are drawn.
        """
        # We expect a batch_size x time x feature_dim layout
        assert (
            tensordict.names[1] == "time"
        ), "Required time dimension to be the second."

        # shape [batch_size, time, num_successor, ] feature can be ragged
        batched_transitions: List[List[List[mi.Transition]]] = tensordict[
            self._env.keys.transitions
        ]
        batched_states: List[List[mi.State]] = tensordict[self._env.keys.state]
        batched_instances: List[List[InstanceType]] = tensordict[
            self._env.keys.instance
        ]

        if len(batched_transitions) == 0:
            raise ValueError(f"Got tensordict with no transitions {tensordict}")
        batch_size = len(batched_transitions)
        time_steps = len(batched_transitions[0])
        if time_steps == 0:
            raise ValueError(f"Got tensordict with no time steps {tensordict}")

        batched_rewards: List[List[torch.Tensor]] = []
        batched_dones: List[List[torch.Tensor]] = []

        for batch_entry in range(batch_size):
            # This is one batch-entry but over time.
            time_transitions = batched_transitions[batch_entry]
            time_states = batched_states[batch_entry]
            time_instances = batched_instances[batch_entry]

            reward_over_time: List[torch.Tensor] = []
            dones_over_time: List[torch.Tensor] = []
            for time_step in range(time_steps):
                # The primary issue here is the lack of homogeneity in `num_transitions`.
                # For each state at a given time step, we have `num_successor` many transitions,
                # but only one `current_state` and one `instance`. This variability exists
                # both across the batch dimension (different batch entries) and the time
                # dimension (different time steps).
                #
                # Because `num_transitions` is not uniform, we cannot efficiently batch
                # these operations. Instead, we must copy `current_state` and `instance`
                # for each transition, which is computationally expensive and results in
                # poor performance.
                num_transitions = len(time_transitions[time_step])
                reward, done = self._env.get_reward_and_done(
                    time_transitions[
                        time_step
                    ],  # all possible transitions for one single state
                    current_states=[time_states[time_step]]
                    * num_transitions,  # copy state for num_transitions
                    instances=[time_instances[time_step]]
                    * num_transitions,  # copy instance for num_transitions
                )
                reward_over_time.append(reward)
                dones_over_time.append(done)
            batched_rewards.append(reward_over_time)
            batched_dones.append(dones_over_time)

        # Return as [batch_size, time, num_successors]
        return batched_rewards, batched_dones


class KeyBasedProvider(RewardDoneProvider):

    def __init__(self, reward_key: NestedKey, done_key: NestedKey):
        self.reward_key = reward_key
        self.done_key = done_key

    @property
    def in_keys(self) -> List[NestedKey]:
        return [self.done_key, self.reward_key]

    def __call__(self, tensordict: TensorDict):
        return tensordict[self.reward_key], tensordict[self.done_key]


class AllActionsValueEstimator(TD0Estimator):
    r"""
    Model-based value estimator that evaluates the agent over all possible actions instead
    of only the sampled action. This assumes that the environment provides all potential successor states
    along with reward and done signals for each possible transition.

    The value estimate is analogous to TD0 but computed for each possible successor state
    and weighted by the agent's transition probability of going to the successor state.

    .. math::
        v(s) = \sum_{s' \in N(s)} \pi(s' \mid s) \cdot td_0(s')
        td_0(s') = r(s,s') + \gamma * v(s') \cdot \text{not_terminated}

    where N(s) is the set of all possible successor states, π(s'|s) is the agent's policy,
    and v(s) is the state value function.

    This estimator provides two values per state: 'value_target' and 'advantage'.
    If `compute_successor_advantages` is True, it also computes an advantage value
    for each possible successor state.

    The user is responsible for providing the input necessary for the `value_network`
    (typically the object embeddings).
    """

    @dataclasses.dataclass
    class _AcceptedKeys(TD0Estimator._AcceptedKeys):
        transition_probabilities: str = ActorCritic.default_keys.probs
        successor_embeddings: str = ActorCritic.default_keys.successor_embeddings
        # advantage for each successor state
        successor_advantage: str = "successor_advantage"

    default_keys = _AcceptedKeys()
    tensor_keys: _AcceptedKeys

    def __init__(
        self,
        value_network: ValueOperator,
        reward_done_provider: RewardDoneProvider,
        gamma: float | torch.Tensor,
        compute_successor_advantages: bool = True,
        keys: _AcceptedKeys = default_keys,
    ):
        """

        :param reward_done_provider: Strategy providing additional rewards and done signals.
        :param gamma: Discount factor.
        :param value_network: Network estimating state values.
        :param compute_successor_advantages: Whether the advantage for each possible action should be computed too,
            as opposed to computing only the advantage for the sampled action.
            If this estimator is used with AllActionsLoss the compute_successor_advantages should be set to true, otherwise
            the estimator can be solely used for a more informed value target for v(s).
        """
        super().__init__(
            value_network=value_network,
            gamma=gamma,
            shifted=True,
            average_rewards=False,
            differentiable=False,
        )
        self._reward_done_provider = reward_done_provider
        self.compute_successor_advantages = compute_successor_advantages
        self._tensor_keys = keys

    def set_keys(self, **kwargs) -> None:
        # Has to be overwritten because instance check using __dict__ does not work.
        for key, value in kwargs.items():
            if not hasattr(self._AcceptedKeys, key):
                raise KeyError(
                    f"{key} is not an accepted tensordict key for advantages"
                )
        if self._tensor_keys is None:
            conf = dataclasses.asdict(self.default_keys)
            conf.update(self.dep_keys)
        else:
            conf = dataclasses.asdict(self._tensor_keys)
        conf.update(kwargs)
        self._tensor_keys = self._AcceptedKeys(**conf)

    def batched_value_estimate(
        self,
        rewards: List[torch.Tensor],
        dones: List[torch.Tensor],
        transition_probabilities: List[torch.Tensor],
        successor_embeddings: List[ObjectEmbedding],  # len(list) == batch_dim
    ) -> Tuple[Tensor, NonTensorStack | None]:
        """
        Expects input to be one batch entry without a time dimension, e.g., the same what
        the environment works on. Shape [batch_size, feature_dim] or [batch_size, num_successor, feature_dim].
        The user is responsible to provide the successor embeddings before calling the estimator.
        Each of rewards, dones, transition_probabilities and successor_embeddings have the exact same shape and should be on the same device.
        :param rewards: The reward for every possible transition.
        :param dones: The done/terminated signal for every possible transition.
        :param transition_probabilities: The probability assigned to each outgoing transition, with transition_probability[i].numel() == number of successors.
        :param successor_embeddings: The embeddings for each successor for each state. They will be only used as input to the value module.
        :return: The batched expected value for each current_state and if `self.compute_successor_advantages` the advantage for each successor.
        """

        # unzip list of tuple into tuple of lists
        flat_reward = torch.cat(rewards)
        flat_done = torch.cat(dones)

        flat_successor_values = torch.cat(
            [
                self.value_network.module(object_embedding)
                for object_embedding in successor_embeddings
            ]
        )
        if flat_reward.device != flat_successor_values.device:
            warnings.warn(
                f"Found mismatching devices. Reward was created at {flat_reward.device} "
                f"but successor values are on {flat_successor_values.device}. "
                f"Moving rewards..."
            )
            flat_reward.to(flat_successor_values.device)
            flat_done.to(flat_successor_values.device)
        if transition_probabilities[0].device != flat_successor_values.device:
            # We don't move because both tensors should be on the accelerator.
            raise RuntimeError(
                f"Found mismatching devices. "
                f"Found transitions probabilities on {transition_probabilities[0].device} "
                f"but successor values are on {flat_successor_values.device}. "
            )

        if flat_successor_values.shape == (*flat_reward.shape, 1):
            flat_successor_values = flat_successor_values.squeeze()
        assert flat_reward.shape == flat_successor_values.shape
        # td0_return_estimate for each possible action.
        flat_targets = (
            flat_reward + self.gamma * flat_successor_values * (~flat_done).int()
        )
        start_indices: List[int] = list(itertools.accumulate(map(len, rewards)))
        # Group flat tensor by source state of outgoing transitions.
        # The value target can be understood as the expected value of the current state
        # if we take a particular action. Just that we evaluate this over all possible
        # actions instead as for only the chosen action.
        # Shape [batch_size, num_successor, ]
        value_targets: Tuple[Tensor, ...] = flat_targets.tensor_split(
            start_indices[:-1], dim=0
        )

        # Expected Value of s = sum over s' in N(s): pi(s' \mid s) * td estimate (s')
        # Shape [batch_size, ]
        expected_value: torch.Tensor = torch.stack(
            [
                value_target.dot(probs)
                for value_target, probs in zip(value_targets, transition_probabilities)
            ]
        )
        if self.compute_successor_advantages:
            # The advantage of for each possible action individually.
            # The td estimate of going to s' - expected value for s.
            #  shape [batch_size, num_successor, 1]
            successor_advantages = as_non_tensor_stack(
                [
                    (value_target - expected_value[i]).unsqueeze(dim=-1)
                    for i, value_target in enumerate(value_targets)
                ]
            )
        else:
            successor_advantages = None

        return expected_value, successor_advantages

    def value_estimate(
        self,
        tensordict: TensorDict,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the value estimate combined over all possible transitions.
        For every time step we do:
            1. Get the reward and done for each possible transition from the environment.
            2. Compute TD0 for single transition R + gamma * v(s') * not_terminated
            3. value_target = sum over successors: TD0-estimate weighted by transition probability
        :param tensordict The rollout-tensordict of shape batch_size x time x feature_dim
            The tensordict has to contain: (...) marks where they typically come from
                - transitions probabilities (agent)
                - successor_embeddings (agen)
                - **in keys of reward_done_provider
        """
        # We expect a batch_size x time x feature_dim layout
        assert (
            tensordict.names[1] == "time"
        ), "Required time dimension to be the second."

        batched_rewards: List[List[torch.Tensor]]
        batched_dones: List[List[torch.Tensor]]
        batched_rewards, batched_dones = self._reward_done_provider(tensordict)
        batched_probs: List[List[torch.Tensor]] = tensordict[
            self._tensor_keys.transition_probabilities
        ]
        # [batch_size, time, num_successor]
        batched_successor_embeddings: List[List[ObjectEmbedding]] = tensordict[
            self._tensor_keys.successor_embeddings
        ]

        time_steps = len(batched_rewards[0])

        # As the tensordict stores information as batch x time we go over each batch entry,
        # this can be inefficient if the time dimension is greater than the batch dimension.

        # Check the device where the input is coming from.
        assert batched_rewards[0][0].device == batched_probs[0][0].device

        with torch.no_grad():  # avoid gradients while computing v(s')
            # [time, batch_size]
            value_targets: List[torch.Tensor] = []
            successor_advantages: List[NonTensorStack | None] = []
            for time_step in range(time_steps):
                batched_value_targets, batched_successor_advantages = (
                    self.batched_value_estimate(
                        rewards=_get_time_step_of(batched_rewards, time_step),
                        dones=_get_time_step_of(batched_dones, time_step),
                        transition_probabilities=_get_time_step_of(
                            batched_probs, time_step
                        ),
                        successor_embeddings=_get_time_step_of(
                            batched_successor_embeddings, time_step
                        ),
                    )
                )
                value_targets.append(batched_value_targets)
                successor_advantages.append(batched_successor_advantages)

        if self.compute_successor_advantages:
            tensordict[self._tensor_keys.successor_advantage] = torch.stack(
                successor_advantages, dim=1
            )

        # value target is expected to be of shape batch x time x 1
        return torch.stack(value_targets, dim=1).unsqueeze(dim=-1)
