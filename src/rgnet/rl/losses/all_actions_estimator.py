import dataclasses
from typing import List, Sequence

import pymimir as mi
import torch
from tensordict import NestedKey, NonTensorStack
from tensordict.nn import TensorDictModule
from torch import Tensor
from torchrl.objectives.value import TD0Estimator

from rgnet.rl import ActorCritic, EmbeddingModule
from rgnet.rl.agents.actor_critic import embed_transition_targets
from rgnet.rl.envs.planning_env import InstanceType, PlanningEnvironment
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack
from rgnet.utils.object_embeddings import ObjectEmbedding


class AllActionsValueEstimator(TD0Estimator):
    @dataclasses.dataclass
    class _AcceptedKeys(TD0Estimator._AcceptedKeys):
        transition_probabilities: NestedKey = ActorCritic.default_keys.probs
        transitions: NestedKey = PlanningEnvironment.default_keys.transitions
        state: NestedKey = PlanningEnvironment.default_keys.state
        instance: NestedKey = PlanningEnvironment.default_keys.instance
        # advantage for each successor state
        individual_advantage: NestedKey = "individual_advantage"

    default_keys = _AcceptedKeys()
    tensor_keys: _AcceptedKeys

    def __init__(
        self,
        *,
        env: PlanningEnvironment,
        embedding_module: EmbeddingModule,
        gamma: float | torch.Tensor,
        value_network: TensorDictModule,
        compute_individual_advantages: bool = False,
        # keys for TD0Estimator
        average_rewards: bool = False,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        skip_existing: bool | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            gamma=gamma,
            value_network=value_network,
            shifted=True,
            average_rewards=average_rewards,
            differentiable=False,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            device=device,
        )
        self._env = env
        self._embedding_module = embedding_module
        self.individual_advantages = compute_individual_advantages

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

    def value_estimate_batch_entry(
        self,
        transitions: Sequence[Sequence[mi.Transition]] | Sequence[mi.Transition],
        current_states: Sequence[mi.State] | mi.State,
        instances: Sequence[InstanceType] | InstanceType,
        transition_probability: Sequence[torch.Tensor] | torch.Tensor,
    ) -> tuple[Tensor, NonTensorStack]:
        gamma = self.gamma.to(self._env.device)

        assert (
            isinstance(transitions, Sequence)
            and isinstance(current_states, Sequence)
            and isinstance(instances, Sequence)
            and isinstance(transition_probability, Sequence)
        )
        time_steps = len(transitions)
        assert time_steps == len(instances) == len(transition_probability)
        successor_embeddings: ObjectEmbedding = embed_transition_targets(
            transitions if time_steps > 0 else [transitions], self._embedding_module
        )
        successor_values: torch.Tensor = self.value_network.module(successor_embeddings)

        # shape = [time_steps]
        expected_value_targets: List[torch.Tensor] = []
        individual_advantages: List[torch.Tensor] = []

        for time in range(time_steps):
            successor_values = successor_values[time]

            reward, done = self._env.get_reward_and_done(
                transitions[time],
                # we can use the same state and instance for all outgoing transitions
                current_states[time : time + 1],
                instances[time : time + 1],
            )
            # td0_return_estimate() but with expected value instead v(current_state)
            value_targets = reward + gamma * successor_values.squeeze() * ~done
            expected_value = value_targets.dot(transition_probability[time])
            expected_value_targets.append(expected_value)
            if self.individual_advantages:
                advantages = value_targets - expected_value
                individual_advantages.append(advantages)

        return torch.stack(expected_value_targets), as_non_tensor_stack(
            individual_advantages
        )

    def value_estimate(
        self,
        tensordict,
        **kwargs,
    ):
        """
        For every time step we do:
            value_target = TD0 estimate weighted by transition probability of agent
        1. Get next states from the transitions
        2. Get the reward and done for each possible transition from the environment
        3. Compute TD0 for single transition R + gamma * v(s') * not_terminated
        """
        # We expect a batch_size x time x ... layout
        assert tensordict.names[-1] == "time"

        batched_transitions = tensordict[self._tensor_keys.transitions]
        batched_states = tensordict[self._tensor_keys.state]
        batched_instances = tensordict[self._tensor_keys.instance]
        batched_probs = tensordict[self._tensor_keys.transition_probabilities]

        # As the tensordict stores information as batch x time we go over each batch entry
        # this can be inefficient if the time dimension is smaller than the batch dimension.
        with torch.no_grad():
            batched_targets = []
            batched_individual_advantages: List[NonTensorStack] = []
            for batch_entry in range(len(batched_transitions)):
                timed_value_targets, advantages = self.value_estimate_batch_entry(
                    batched_transitions[batch_entry],
                    batched_states[batch_entry],
                    batched_instances[batch_entry],
                    batched_probs[batch_entry],
                )
                batched_targets.append(timed_value_targets)
                if self.individual_advantages:
                    batched_individual_advantages.append(advantages)

        if self.individual_advantages:
            tensordict[self.tensor_keys.individual_advantage] = torch.stack(
                batched_individual_advantages
            )

        # value target is expected to be of shape batch x time x 1
        return torch.stack(batched_targets).unsqueeze(dim=-1)
