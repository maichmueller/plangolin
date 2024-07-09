from test.fixtures import medium_blocks, small_blocks
from test.rl.envs.test_state_space_env import _test_rollout_soundness
from typing import List

import mockito
import pytest
import torch
from tensordict import NonTensorData

from rgnet.rl import Agent, EmbeddingModule
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, non_tensor_to_list


@pytest.fixture
def embedding_mock(hidden_size=2):

    def random_embeddings(states: List | NonTensorWrapper):
        states = non_tensor_to_list(states)
        batch_size = len(states)
        return torch.randn(size=(batch_size, hidden_size))

    mock = mockito.mock(
        {"hidden_size": hidden_size, "__call__": random_embeddings},
        strict=True,
        spec=EmbeddingModule,
    )
    # some torchrl/tensordict asks for the named modules. Sadly I don't know how to
    # use the original Module implemented method.
    mockito.when(mock).named_modules(...).thenReturn([("", mock)])
    return mock


@pytest.fixture
@pytest.mark.parametrize("embedding_mock", [2], indirect=True)
def agent(small_blocks, embedding_mock):
    _, domain, _ = small_blocks
    return Agent(embedding_mock)


@pytest.fixture
def env(small_blocks, batch_size):
    space, _, _ = small_blocks
    return ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)


@pytest.fixture
def medium_env(medium_blocks, batch_size):
    space, _, _ = medium_blocks
    return ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)


def test_init(agent):
    assert agent._hidden_size == 2
    assert agent._embedding_module is not None
    assert agent.actor_net is not None
    assert agent.value_operator is not None
    assert agent.probabilistic_module is not None


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("space_fixture", ["small_blocks", "medium_blocks"])
@pytest.mark.parametrize("rollout_length", [5, 20])
def test_as_policy(batch_size, agent, space_fixture, rollout_length, request):
    space, _, _ = request.getfixturevalue(space_fixture)
    environment = ExpandedStateSpaceEnv(
        space, batch_size=torch.Size([batch_size]), seed=42
    )
    policy = agent.as_td_module(
        environment.keys.state, environment.keys.transitions, environment.keys.action
    )
    rollout = environment.rollout(
        rollout_length, policy=policy, break_when_any_done=False
    )

    env_keys = environment.keys
    expected_keys = {
        env_keys.action,
        agent.current_embedding,
        env_keys.done,
        env_keys.goals,
        env_keys.transitions,
        env_keys.state,
        env_keys.terminated,
        Agent.log_probs,
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
            log_prob_tensor = rollout.get(Agent.log_probs)[batch_idx][time_step]
            assert log_prob_tensor.numel() == 1
            # log of a probability is in (-infty, 0]
            assert (log_prob_tensor <= 0).all()


@pytest.mark.parametrize("embedding_mock", [10], indirect=True)
def test_policy_preparation(embedding_mock):
    batch_size = 3  # hardcoded for this test

    hidden_size = embedding_mock.hidden_size

    def actor_mock(tensor):
        _batch_size = tensor.shape[0]
        return torch.ones(size=(_batch_size, 1), dtype=tensor.dtype)

    agent = Agent(embedding_mock)

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
