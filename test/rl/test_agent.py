from test.fixtures import small_blocks
from typing import List

import mockito
import pytest
import torch
import torch.nn.functional as F
from tensordict import NonTensorData, NonTensorStack, TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import ValueEstimators
from torchrl.objectives.value import TD0Estimator

from rgnet import HeteroGraphEncoder
from rgnet.rl import Agent, EmbeddingModule
from rgnet.rl.agent import PolicyPreparationModule
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


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
def agent(small_blocks, embedding_mock, value_estimator_type=None):
    _, domain, _ = small_blocks
    estimator = value_estimator_type or ValueEstimators.TD0
    return Agent(embedding_mock, value_estimator_type=estimator)


@pytest.fixture
def environment(small_blocks, batch_size):
    space, _, _ = small_blocks
    # TODO use mocking for all side-effects
    environment = ExpandedStateSpaceEnv(
        space, batch_size=torch.Size([batch_size]), seed=42
    )
    return environment


def test_init(agent):
    assert agent._hidden_size == 2
    assert agent._embedding_module is not None
    assert agent.embedding_td_module is not None
    assert agent.prob_actor is not None
    assert agent.value_operator is not None
    assert agent.action_selector is not None
    assert agent.policy is not None
    assert agent.loss is not None


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "value_estimator_type",
    [ValueEstimators.TD0, ValueEstimators.TD1, ValueEstimators.GAE],
)
def test_manual_agent_use(batch_size, agent, environment, value_estimator_type):
    td = environment.reset()

    agent.embedding_td_module(td)

    c_key = agent.default_keys.current_embedding
    assert c_key in td
    assert isinstance(td.get(c_key), torch.Tensor)
    assert td.get(c_key).shape[0] == batch_size  # first dimension ist batch_size

    agent.prob_actor(td)

    assert "logits" in td
    assert isinstance(td.get("logits"), torch.Tensor)
    assert td.get("logits").shape[0] == batch_size  # first dimension ist batch_size
    action_idx = agent.default_keys.action_idx
    assert action_idx in td
    assert isinstance(td.get(action_idx), torch.Tensor)
    assert td.get(action_idx).shape == (batch_size,)
    nbr_transitions = [
        len(batched_transitions)
        for batched_transitions in td[agent.default_keys.transitions]
    ]
    assert (
        0 >= t.item() <= nbr_t for (nbr_t, t) in zip(nbr_transitions, td[action_idx])
    )

    agent.action_selector(td)
    action_key = agent.default_keys.action
    assert action_key in td
    assert isinstance(td.get(action_key), NonTensorStack)
    assert td.get(action_key).batch_size == (batch_size,)
    actions = td[action_key]
    transitions_batch = td[agent.default_keys.transitions]
    for i, transitions in enumerate(transitions_batch):
        i_th_action_idx = td[action_idx][i].item()
        assert actions[i] == transitions[i_th_action_idx]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "value_estimator_type",
    [ValueEstimators.TD0, ValueEstimators.TD1, ValueEstimators.GAE],
)
def test_as_policy(batch_size, agent, environment, value_estimator_type):
    environment.set_seed(42)
    rollout_length = 5
    rollout = environment.rollout(
        rollout_length, policy=agent.policy, break_when_any_done=False
    )
    keys = environment.default_keys
    expected_root_keys = [
        keys.action,
        keys.done,
        keys.goals,
        keys.state,
        keys.terminated,
        keys.transitions,
    ]
    # Assert that all shapes match batch_size and rollout_length
    for key in expected_root_keys:
        assert key in rollout
        val = rollout.get(key)
        if isinstance(val, torch.Tensor):
            assert val.shape == (batch_size, rollout_length, 1)
        else:
            assert isinstance(val, NonTensorStack)
            assert val.batch_size == (batch_size, rollout_length)

    # TODO test soundness of produced rollout


@pytest.mark.parametrize("embedding_mock", [10], indirect=True)
def test_policy_preparation(embedding_mock):
    batch_size = 5  # hardcoded for this test

    def actor_mock(tensor):
        batch_size = tensor.shape[0]
        return torch.ones(size=(batch_size, 1), dtype=tensor.dtype)

    actor_net_mock = mockito.mock({"__call__": actor_mock})

    # Only the target is used and passed to the embeddings module so the value doesn't
    # matter
    transition_mock = mockito.mock({"target": "X"})

    policy_preparation = PolicyPreparationModule(embedding_mock, actor_net_mock)
    assert policy_preparation.in_keys == [
        Agent.default_keys.current_embedding,
        Agent.default_keys.transitions,
    ]
    transitions = as_non_tensor_stack(
        [
            [transition_mock] * 3,
            [transition_mock] * 4,
            [transition_mock] * 2,
            [transition_mock] * 3,
            [transition_mock] * 1,
        ]
    )
    td = TensorDict(
        {
            policy_preparation.in_keys[0]: embedding_mock([None] * batch_size),
            policy_preparation.in_keys[1]: transitions,
        },
        batch_size=batch_size,
    )

    out_td = policy_preparation(td)
    assert out_td is td, "PolicyPreparationModule should not create new tensordicts"
    assert all(key in out_td for key in policy_preparation.out_keys)
    logits = out_td.get(policy_preparation.out_keys[0])
    # We get 1 (from the mock) for each pair and 0s as padding
    # softmax is 1 / (number of successors)
    expected_logits = torch.tensor(
        [
            [1.0 / 3.0] * 3 + [0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0, 0.0],
            [1.0 / 3.0] * 3 + [0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(logits, expected_logits)

    mockito.verify(embedding_mock, times=2).__call__(...)
    # At most one call for every element in the batch_size
    mockito.verify(actor_net_mock, atmost=batch_size).__call__(...)

    # Assert that another call to the module where the output is already present
    # skips the computation
    mockito.forget_invocations(actor_net_mock)
    policy_preparation(td)
    mockito.verify(actor_net_mock, times=0).__call__(...)

    # Assert mismatching between number of current_embeddings and transitions
    illegal_td = TensorDict(
        {
            policy_preparation.in_keys[0]: embedding_mock([None] * batch_size),
            policy_preparation.in_keys[1]: NonTensorData(
                [transition_mock] * 3, batch_size=(1,)
            ),
        },
    )
    with pytest.raises(AssertionError):
        policy_preparation(illegal_td)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_loss(small_blocks, environment, batch_size):
    _, domain, _ = small_blocks
    embedding = EmbeddingModule(
        HeteroGraphEncoder(domain), hidden_size=2, num_layer=1, aggr="sum"
    )
    estimator = ValueEstimators.TD0
    agent = Agent(
        embedding_module=embedding,
        value_estimator_type=estimator,
        value_estimator_kwargs={"gamma": 0.9},
        loss_kwargs={
            "entropy_bonus": False,
            "loss_critic_type": "l2",
            "reduction": "mean",
            "clip_value": None,
        },
    )
    agent.loss.make_value_estimator(ValueEstimators.TD0, gamma=0.9)
    agent.loss.loss_critic_type = "l2"
    td = environment.reset()
    agent.policy(td)
    environment.step(td)
    agent.embedding_td_module(td.get("next"))
    agent.loss.value_estimator(td)
    td["advantage"] = td["advantage"].detach()
    loss_td = agent.loss(td)
    assert isinstance(loss_td, TensorDict)
    assert ["loss_critic", "loss_objective"] == loss_td.sorted_keys

    # expected loss for the critic
    value_network = TensorDictModule(
        lambda x: 1 / 0,  # raise exception
        in_keys=["obs"],
        out_keys=["state_value"],
    )
    estimator = TD0Estimator(
        gamma=0.9,
        value_network=value_network,
        average_rewards=False,
        skip_existing=True,
    )
    td_clone = td.select(
        "state_value", ("next", "state_value"), ("next", "reward"), ("next", "done")
    )
    estimator(td_clone)
    assert torch.allclose(td["value_target"], td_clone["value_target"])
    assert torch.allclose(td["advantage"], td_clone["advantage"])

    # the loss is reduced over the batch by the 'reduction' argument to the loss
    expected_loss_critic = F.mse_loss(
        td["state_value"], td_clone["value_target"], reduction="mean"
    )
    assert torch.allclose(expected_loss_critic, loss_td["loss_critic"])
