from test.fixtures import problem_setup

import pytest
import torch
from tensordict import NonTensorStack

from rgnet import HeteroGraphEncoder
from rgnet.rl import Agent, EmbeddingModule
from rgnet.rl.envs import ExpandedStateSpaceEnv


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_manual_agent_use(batch_size):
    space, domain, _ = problem_setup("blocks", "small")
    # TODO use mocking for all side-effects
    embedding = EmbeddingModule(
        HeteroGraphEncoder(domain), hidden_size=2, num_layer=1, aggr="sum"
    )
    agent = Agent(embedding)
    environment = ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]))
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
