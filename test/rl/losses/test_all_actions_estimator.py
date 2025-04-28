from test.fixtures import (  # noqa: F401
    embedding_mock,
    medium_blocks,
    multi_instance_env,
    small_blocks,
)
from typing import List, Tuple

import mockito
import pytest
import torch
from tensordict import LazyStackedTensorDict, NonTensorStack, TensorDict
from torch import tensor
from torchrl.modules import ValueOperator

from rgnet.rl.agents import ActorCritic
from rgnet.rl.envs import MultiInstanceStateSpaceEnv, PlanningEnvironment
from rgnet.rl.losses import AllActionsValueEstimator
from rgnet.rl.losses.all_actions_estimator import (
    EnvironmentBasedRewardProvider,
    KeyBasedProvider,
)
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack
from rgnet.utils.object_embeddings import ObjectEmbedding, ObjectPoolingModule


def _object_embedding_of(single_dim_embedding: List[float]):
    return ObjectEmbedding(
        torch.tensor(single_dim_embedding, dtype=torch.float).view(-1, 1, 1),
        padding_mask=torch.ones((len(single_dim_embedding), 1), dtype=torch.bool),
    ).to_tensordict()


class TestAllActionsValueEstimator:
    """
    Covered test scenarios:
    - compute_successor_advantages = True
    - End to end: call as normal TD0Estimator -> check if values are correct
    - Environment-based provider
        - compute_successor_advantages = False
    Not covered:
        - One time step vs multiple
        - Device sensible test
            - successor_embeddings, probs and reward / done
            - probs will be computed on GPU
            - reward / done could come from CPU
    """

    GAMMA = 0.9
    actor_keys = ActorCritic.default_keys
    env_keys = PlanningEnvironment.default_keys

    @pytest.fixture
    def critic_mock(self):
        class Mock(torch.nn.Module):
            # Use the aggregated object embeddings directly as value (assumes hidden_dim=1).
            pooling_module = ObjectPoolingModule()

            def forward(self, embeddings: ObjectEmbedding | TensorDict):
                embedding_tensor = self.pooling_module(embeddings)
                embedding_tensor.requires_grad = True
                return embedding_tensor

        critic = ValueOperator(Mock(), in_keys=[self.actor_keys.current_embedding])
        mockito.spy2(critic.forward)
        return critic

    # noinspection PyTypeChecker
    @pytest.fixture
    def rollout_data(self) -> Tuple[TensorDict, tensor, tensor, NonTensorStack]:
        """
        batch_size = 2
        time steps = 2

        Done in the second time-step of the second batch entry.
        Agent always chose 0-th action.
        We use hidden_size=1 for object embeddings and one object per state.
        Hence, the object-embedding = state embedding = state value.

        tensordict keys:
            'current_embedding',
            'successor_embeddings',
            'action',
            'probs',
            'log_probs',
            'idx_in_space',
            'all_dones',
            'all_rewards',
            next:
                'terminated',
                'done', # all_dones[action]
                'reward',
                'current_embedding'
        .. mermaid::
            stateDiagram-v2
                [*] --> 0
                0 --> 1: Chosen
                0 --> 2
                0 --> 3
                [*] --> 4
                4 --> 5: Chosen
                4 --> 6
                1 --> 5: Chosen
                1 --> 6
                5 --> 4: Chosen
                5 --> [*]
                note left of 5: Goal state
        """
        current_embedding = _object_embedding_of([0.0, 4.0])
        split_successor_embeddings = [
            _object_embedding_of([1.0, 2.0, 3.0]),
            _object_embedding_of([5.0, 6.0]),
        ]
        first_step = TensorDict(
            {
                self.actor_keys.current_embedding: current_embedding,
                self.actor_keys.successor_embeddings: as_non_tensor_stack(
                    split_successor_embeddings
                ),
                self.actor_keys.probs: as_non_tensor_stack(
                    [
                        tensor([0.9, 0.1, 0.0], requires_grad=True),
                        tensor([0.5, 0.5], requires_grad=True),
                    ]
                ),
                self.actor_keys.log_probs: tensor([0.9, 0.5]).log(),
                self.env_keys.action: tensor([0, 0]),
                "all_dones": as_non_tensor_stack(
                    [tensor([False, False, False]), tensor([False, False])]
                ),
                "all_rewards": as_non_tensor_stack(
                    [tensor([-1.0, -1.0, -1.0]), tensor([-1.0, -1.0])]
                ),
            },
            batch_size=torch.Size((2,)),
        )
        first_step["next"] = TensorDict(
            {
                self.env_keys.done: tensor([False, False]).unsqueeze(dim=-1),
                self.env_keys.reward: tensor([-1.0, -1.0]).unsqueeze(dim=-1),
                self.actor_keys.current_embedding: tensor([1.0, 5.0]).view((2, 1, 1)),
            },
            batch_size=torch.Size((2,)),
        )
        second_step_embedding = _object_embedding_of([1.0, 5.0])
        second_step_successor_embedding = [
            _object_embedding_of([5.0, 6.0]),
            _object_embedding_of([4.0]),
        ]
        second_step = TensorDict(
            {
                self.actor_keys.current_embedding: second_step_embedding,
                self.actor_keys.successor_embeddings: as_non_tensor_stack(
                    second_step_successor_embedding
                ),
                self.actor_keys.probs: as_non_tensor_stack(
                    [
                        tensor([0.5, 0.5], requires_grad=True),
                        tensor([1.0], requires_grad=True),
                    ]
                ),
                self.actor_keys.log_probs: tensor([0.5, 1.0]).log(),
                self.env_keys.action: tensor([0, 0]),
                "all_dones": as_non_tensor_stack(
                    [tensor([False, False]), tensor([True])]
                ),
                "all_rewards": as_non_tensor_stack(
                    [tensor([-1.0, -1.0]), tensor([0.0])]
                ),
            },
            batch_size=torch.Size((2,)),
        )
        second_step["next"] = TensorDict(
            {
                self.env_keys.done: tensor([False, True]).unsqueeze(dim=-1),
                self.env_keys.reward: tensor([-1.0, 0.0]).unsqueeze(dim=-1),
                self.actor_keys.current_embedding: _object_embedding_of([5.0, 4.0]),
            },
            batch_size=torch.Size((2,)),
        )
        # The expected value target for current states using all actions.
        # [batch_size, time, 1]
        expected_targets = torch.stack(
            # e.g., for the first step of first batch = 0.9 * (-1 + self.GAMMA * 1) + 0.1 * (-1 + self.GAMMA * 2)
            # 0.0 for done state 1.0 * (0.0)
            [tensor([-0.01, 3.95]), tensor([3.95, 0.0])]
        ).unsqueeze(dim=-1)
        td = LazyStackedTensorDict.maybe_dense_stack([first_step, second_step], dim=1)
        td.refine_names(..., "time")
        obj_embedding = ObjectEmbedding.from_tensordict(
            td[self.actor_keys.current_embedding]
        )

        advantage = expected_targets - ObjectPoolingModule()(obj_embedding)

        # The advantage for each possible transition.
        # Shape is [batch_size, time, num_successors, ]
        first_step_advantages = [
            tensor([-1 + self.GAMMA * 1, -1 + self.GAMMA * 2, -1 + self.GAMMA * 3]),
            tensor([-1 + self.GAMMA * 5, -1 + self.GAMMA * 6]),
        ]
        first_step_advantages = as_non_tensor_stack(
            [
                individual_target.unsqueeze(dim=-1) - expected_targets[idx, 0, :]
                for idx, individual_target in enumerate(first_step_advantages)
            ]
        )
        second_step_advantages = [
            tensor([-1 + self.GAMMA * 5, -1 + self.GAMMA * 6]),
            tensor([0.0]),
        ]
        second_step_advantages = as_non_tensor_stack(
            [
                individual_target.unsqueeze(dim=-1) - expected_targets[idx, 1, :]
                for idx, individual_target in enumerate(second_step_advantages)
            ]
        )
        successor_targets = torch.stack([first_step_advantages, second_step_advantages])
        return td, expected_targets, advantage, successor_targets

    def test_value_estimate(
        self,
        critic_mock,
        rollout_data,
    ):
        in_data, expected_targets, expected_advantage, expected_successor_advantage = (
            rollout_data
        )
        estimator = AllActionsValueEstimator(
            critic_mock,
            KeyBasedProvider("all_rewards", "all_dones"),
            gamma=self.GAMMA,
            shifted=True,
            compute_successor_advantages=True,
        )
        result = estimator(in_data)
        expected_keys = [
            estimator.tensor_keys.value,
            estimator.tensor_keys.value_target,
            estimator.tensor_keys.advantage,
            estimator.tensor_keys.successor_advantage,
        ]
        assert all(key in result.keys() for key in expected_keys)
        assert torch.allclose(
            result[estimator.tensor_keys.value_target], expected_targets
        )
        assert torch.allclose(
            result[estimator.tensor_keys.advantage], expected_advantage
        )
        for batched_real, batched_expected in zip(
            result[estimator.tensor_keys.successor_advantage],
            expected_successor_advantage.tolist(),
        ):
            assert all(
                torch.allclose(real, expected)
                for (real, expected) in zip(batched_real, batched_expected)
            )

    @pytest.mark.parametrize(
        "multi_instance_env",
        [dict(spaces=["small_blocks", "medium_blocks"], batch_size=5)],
        indirect=True,
    )
    @pytest.mark.parametrize("hidden_size", [4])
    def test_env_based_provider(
        self,
        multi_instance_env: MultiInstanceStateSpaceEnv,
        embedding_mock,
        hidden_size,
    ):
        agent = ActorCritic(
            hidden_size=embedding_mock.hidden_size,
            embedding_module=embedding_mock,
            add_successor_embeddings=True,
        )
        env_keys = multi_instance_env.keys
        policy = agent.as_td_module(
            env_keys.state, env_keys.transitions, env_keys.action, add_probs=True
        )

        provider = EnvironmentBasedRewardProvider(multi_instance_env)
        estimator = AllActionsValueEstimator(
            value_network=agent.value_operator,
            reward_done_provider=provider,
            gamma=0.9,
            compute_successor_advantages=False,
        )

        rollout = multi_instance_env.rollout(
            max_steps=3,
            policy=policy,
            break_when_any_done=False,
        )
        # Embedding of the last state in next required for TD(0) estimator
        # this should not be done in a non-test case.
        rollout["next"]["current_embedding"] = rollout["current_embedding"]
        estimate: TensorDict = estimator(rollout)

        for key in [
            estimator.tensor_keys.value,
            estimator.tensor_keys.value_target,
            estimator.tensor_keys.advantage,
        ]:
            result = estimate[key]
            assert isinstance(result, torch.Tensor)
            assert result.shape == (5, 3, 1)

        assert estimator.tensor_keys.successor_advantage not in estimate.keys()
