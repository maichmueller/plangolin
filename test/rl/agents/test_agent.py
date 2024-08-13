from test.fixtures import embedding_mock, medium_blocks, small_blocks
from test.rl.envs.test_state_space_env import _test_rollout_soundness
from typing import List

import mockito
import pytest
import torch
from tensordict import NonTensorData, NonTensorStack

from rgnet.rl import ActorCritic
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import non_tensor_to_list


@pytest.fixture
def agent(embedding_mock):
    return ActorCritic(embedding_mock)


@pytest.fixture
def env(small_blocks, batch_size):
    space, _, _ = small_blocks
    return ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)


@pytest.fixture
def medium_env(medium_blocks, batch_size):
    space, _, _ = medium_blocks
    return ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)


@pytest.mark.parametrize("hidden_size", [3])
def test_init(agent, hidden_size):
    assert agent._hidden_size == hidden_size
    assert agent._embedding_module is not None
    assert agent.actor_net is not None
    assert agent.value_operator is not None
    assert agent.probabilistic_module is not None


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("hidden_size", [3])
@pytest.mark.parametrize("space_fixture", ["small_blocks", "medium_blocks"])
@pytest.mark.parametrize("rollout_length", [5, 20])
def test_as_policy(batch_size, agent, space_fixture, rollout_length, request):
    space, _, _ = request.getfixturevalue(space_fixture)
    environment = ExpandedStateSpaceEnv(
        space, batch_size=torch.Size([batch_size]), seed=42
    )
    policy = agent.as_td_module(
        environment.keys.state,
        environment.keys.transitions,
        environment.keys.action,
        add_probs=True,
    )
    rollout = environment.rollout(
        rollout_length, policy=policy, break_when_any_done=False
    )

    env_keys = environment.keys
    expected_keys = {
        env_keys.action,
        agent.keys.current_embedding,
        env_keys.done,
        env_keys.goals,
        env_keys.transitions,
        env_keys.state,
        env_keys.terminated,
        agent.keys.log_probs,
        agent.keys.probs,
    }
    _test_rollout_soundness(
        space,
        rollout,
        batch_size,
        rollout_length,
        environment._initial_state,
        expected_keys,
    )

    # test that the probability tensors have no probability-mass in the padded regions.
    for time_step in range(0, rollout_length):
        for batch_idx in range(0, batch_size):
            log_prob_tensor = rollout.get(agent.keys.log_probs)[batch_idx][time_step]
            assert log_prob_tensor.numel() == 1
            assert log_prob_tensor.requires_grad
            # log of a probability is in (-infty, 0]
            assert (log_prob_tensor <= 0).all()

            probs = non_tensor_to_list(
                rollout.get(agent.keys.probs)[batch_idx][time_step]
            )
            # assert that all values are between 0 and 1
            assert torch.all(probs >= 0.0)
            assert torch.all(probs <= 1.0)
            # assert that there are as many probs as transitions
            assert probs.numel() == len(
                rollout[env_keys.transitions][batch_idx][time_step]
            )


@pytest.mark.parametrize("hidden_size", [3])
def test_policy_preparation(embedding_mock, hidden_size):
    batch_size = 3  # hardcoded for this test

    def actor_mock(tensor):
        _batch_size = tensor.shape[0]
        return torch.ones(size=(_batch_size, 1), dtype=tensor.dtype)

    agent = ActorCritic(embedding_mock)

    mockito.when(agent.actor_net).forward(...).thenAnswer(actor_mock)

    current_embeddings = torch.rand(size=(batch_size, hidden_size))

    successor_embeddings = (
        torch.rand(size=(3, hidden_size)),
        torch.rand(size=(4, hidden_size)),
        torch.rand(size=(2, hidden_size)),
    )

    batched_probabilities = agent._actor_probs(current_embeddings, successor_embeddings)

    assert isinstance(batched_probabilities, List)
    assert all(isinstance(t, torch.Tensor) for t in batched_probabilities)
    assert len(batched_probabilities) == batch_size
    # We get 1 (from the mock) for each pair and 0s as padding
    # softmax is 1 / (number of successors)
    expected_batched_probs = [
        torch.tensor([1.0 / 3.0] * 3),
        torch.tensor([0.25, 0.25, 0.25, 0.25]),
        torch.tensor([0.5, 0.5]),
    ]
    assert all(
        torch.allclose(probs, expected_probs)
        for probs, expected_probs in zip(batched_probabilities, expected_batched_probs)
    )
    mockito.verify(embedding_mock, times=0).__call__(...)
    # At most one call for every element in the batch_size
    mockito.verify(agent.actor_net, times=batch_size).forward(...)

    # Assert mismatching between number of current_embeddings and transitions
    with pytest.raises(AssertionError):
        agent._actor_probs(
            embedding_mock([None] * batch_size),
            NonTensorData([None] * 3, batch_size=(1,)),
        )


@pytest.mark.parametrize("hidden_size", [3])
@pytest.mark.parametrize("batch_size", [2])
def test_probabilities_require_grad(agent, env, hidden_size, batch_size):
    """As of the 13.08.2024 NonTensorData will remove any gradients by calling .data
    on the tensor. This test ensures that the gradients are kept for the transition
    probabilities, which don't share a uniform shape across the batch and time dimension
    and therefore have to be wrapped in NonTensorData.
    This test requires torchrl_patches.py to be applied before running the test"""
    rollout = env.reset()
    policy = agent.as_td_module(
        env.keys.state, env.keys.transitions, env.keys.action, add_probs=True
    )
    out = policy(rollout)
    probs_non_tensor_stack = out.get(agent.keys.probs)
    assert isinstance(probs_non_tensor_stack, NonTensorStack)
    probs = non_tensor_to_list(probs_non_tensor_stack)
    assert all(p.requires_grad for p in probs)
    assert len(probs) == batch_size
