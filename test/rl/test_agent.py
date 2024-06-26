from test.fixtures import small_blocks

import mockito
import pytest
import torch
import torch.nn.functional as F
from tensordict import NonTensorStack, TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import ValueEstimators
from torchrl.objectives.value import TD0Estimator

from rgnet import HeteroGraphEncoder
from rgnet.rl import Agent, EmbeddingModule
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack


@pytest.fixture
def agent(small_blocks, value_estimator_type=None):
    _, domain, _ = small_blocks
    # TODO use mocking for all side-effects
    embedding = EmbeddingModule(
        HeteroGraphEncoder(domain), hidden_size=2, num_layer=1, aggr="sum"
    )
    estimator = value_estimator_type or ValueEstimators.TD0
    return Agent(embedding, value_estimator_type=estimator)


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
    assert agent.embedding_td_module is not None
    assert agent.actor_net is not None
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

    c_key = (agent.default_keys.current_embedding,)
    succ_key = (agent.default_keys.successor_embedding,)
    assert c_key in td and succ_key in td
    assert isinstance(td.get(c_key), torch.Tensor)
    assert isinstance(td.get(succ_key), NonTensorStack)
    assert td.get(c_key).shape[0] == batch_size  # first dimension ist batch_size
    assert td.get(succ_key).batch_size == (batch_size,)
    assert td[succ_key][0] is not None

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


def test_policy_function(agent):
    hidden_size = 10

    def actor_mock(tensor):
        batch_size = tensor.shape[0]
        return torch.ones(size=(batch_size, 1), dtype=tensor.dtype)

    def verify(result):
        assert result.shape == (5, 4)
        # We get 1 (from the mock) for each pair and 0s as padding
        # softmax is 1 / (number of successors)
        expected = torch.tensor(
            [
                [1.0 / 3.0] * 3 + [0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.0, 0.0],
                [1.0 / 3.0] * 3 + [0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        assert torch.allclose(result, expected)

    with mockito.patch(agent.actor_net, actor_mock):

        current_embeddings = torch.rand(5, hidden_size)
        successor_embeddings = [
            torch.rand(3, hidden_size),
            torch.rand(4, hidden_size),
            torch.rand(2, hidden_size),
            torch.rand(3, hidden_size),
            torch.rand(1, hidden_size),
        ]
        logits = agent._policy_function(current_embeddings, successor_embeddings)
        verify(logits)

        # Wrap successor_embeddings in a NonTensorStack
        logits = agent._policy_function(
            current_embeddings, as_non_tensor_stack(successor_embeddings)
        )
        verify(logits)

        # Assert mismatching between number of current_embeddings and successors throws
        with pytest.raises(AssertionError):
            agent._policy_function(torch.rand(2, 1), [torch.rand(1, 1)])


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
