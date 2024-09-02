import dataclasses
from typing import List

import pymimir as mi
import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import Tensor
from torchrl.objectives.value import TD0Estimator

from rgnet.rl import ActorCritic, EmbeddingModule
from rgnet.rl.agents.actor_critic import embed_transition_targets
from rgnet.rl.envs.planning_env import PlanningEnvironment


class AllActionsValueEstimator(TD0Estimator):

    def __init__(
        self,
        *,
        env: PlanningEnvironment,
        embedding_module: EmbeddingModule,
        gamma: float | torch.Tensor,
        value_network: TensorDictModule,
        average_rewards: bool = False,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        skip_existing: bool | None = None,
        device: torch.device | None = None
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

    @dataclasses.dataclass
    class _AcceptedKeys(TD0Estimator._AcceptedKeys):
        transition_probabilities: NestedKey = ActorCritic.default_keys.probs
        transitions: NestedKey = PlanningEnvironment.default_keys.transitions
        state: NestedKey = PlanningEnvironment.default_keys.state
        instance: NestedKey = PlanningEnvironment.default_keys.instance

    def value_estimate_batch_entry(
        self,
        timed_transitions: List[List[mi.Transition]],
        timed_current_states: List[mi.State],
        timed_instances: List,
        timed_transition_probability,
    ) -> Tensor:
        gamma = self.gamma.to(self._env.device)
        time_steps = len(timed_transitions)
        assert time_steps == len(timed_instances) == len(timed_transition_probability)
        successor_embeddings: tuple[Tensor, ...] = embed_transition_targets(
            timed_transitions, self._embedding_module
        )

        # shape = [time_steps]
        expected_value_targets: List[torch.Tensor] = []

        for time in range(time_steps):
            successor_values = self.value_network.module(successor_embeddings[time])

            reward, done = self._env.get_reward_and_done(
                timed_transitions[time],
                # we can use the same state and instance for all outgoing transitions
                timed_current_states[time : time + 1],
                timed_instances[time : time + 1],
            )
            # td0_return_estimate()
            value_targets = reward + gamma * successor_values.squeeze() * ~done
            expected_value = value_targets.dot(timed_transition_probability[time])
            expected_value_targets.append(expected_value)

        return torch.stack(expected_value_targets)

    def value_estimate(
        self,
        tensordict,
        target_params: TensorDictBase | None = None,
        next_value: torch.Tensor | None = None,
        **kwargs
    ):
        """
        For every time step we do:
            value_target = TD0 estimate weighted by transition probability of agent
        1. Get next states from the transitions
        2. Get the reward and done for each possible transition from the environment
        3. Compute TD0 for single transition R + gamma * v(s') * not_terminated
        #not_terminated = (~terminated).int()

        #advantage = reward + gamma * not_terminated * next_state_value
        """
        batched_transitions = tensordict[self._tensor_keys.transitions]
        batched_states = tensordict[self._tensor_keys.state]
        batched_instances = tensordict[self._tensor_keys.instance]
        batched_probs = tensordict[self._tensor_keys.transition_probabilities]

        batched_targets = []
        for batch_entry in range(len(batched_transitions)):
            timed_value_targets = self.value_estimate_batch_entry(
                batched_transitions[batch_entry],
                batched_states[batch_entry],
                batched_instances[batch_entry],
                batched_probs[batch_entry],
            )
            batched_targets.append(timed_value_targets)

        # value target is expected to be of shape batch x time x 1
        return torch.stack(batched_targets).unsqueeze(dim=-1)
