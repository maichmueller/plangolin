from test.fixtures import fresh_drive

import mockito
import torch.optim
from torchrl.modules import ValueOperator
from torchrl.objectives import ValueEstimators

from rgnet import HeteroGNN
from rgnet.rl import ActorCritic, ActorCriticLoss
from rgnet.rl.thundeRL.collate import collate_fn
from rgnet.rl.thundeRL.lightning_adapter import LightningAdapter


def test_training_step(fresh_drive):

    BATCH_SIZE = 5
    GAMMA = 0.9

    batch = [fresh_drive[i] for i in range(BATCH_SIZE)]
    batch[0].done = torch.full_like(batch[0].done, fill_value=True)
    batch[0].reward = torch.full_like(batch[0].reward, fill_value=0.0)
    batched_tuple = collate_fn(batch)

    def gnn_forward(x_dict, edge_index_dict, batch_dict):
        batch_size = batch_dict["obj"].max().item() + 1
        if batch_size == BATCH_SIZE:
            return torch.arange(BATCH_SIZE, dtype=torch.float, requires_grad=True).view(
                -1, 1
            )
        return torch.ones(
            size=(batch_size, 1),
            dtype=torch.float,
            requires_grad=True,
        )

    gnn_mock = mockito.mock(HeteroGNN)
    mockito.when(gnn_mock).__call__(...).thenAnswer(gnn_forward)

    value_net_mock = torch.nn.Module()

    def value_net_forward(embedding: torch.Tensor):
        return embedding

    mockito.when(value_net_mock).forward(...).thenAnswer(value_net_forward)

    operator_mock = ValueOperator(
        value_net_mock,
        [ActorCritic.default_keys.current_embedding],
        [ActorCritic.default_keys.state_value],
    )

    log_probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float).log()

    actor_critic_mock = mockito.mock(
        {
            "embedded_forward": lambda x, y: (
                torch.zeros((5,), dtype=torch.long),
                log_probs,
            )
        },
        spec=ActorCritic,
    )

    # return based on the length of the input tensor

    loss = ActorCriticLoss(operator_mock, reduction="mean", loss_critic_type="l2")
    loss.make_value_estimator(ValueEstimators.TD0, gamma=GAMMA)
    optimizer_mock = mockito.mock(spec=torch.optim.Optimizer)

    adapter = LightningAdapter(
        gnn_mock, actor_critic_mock, loss=loss, optim=optimizer_mock
    )

    loss = adapter.training_step(batched_tuple)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    # expected_loss
    # critic_loss
    values = torch.arange(BATCH_SIZE)
    # rewards + gamma * next_values
    targets = torch.full((BATCH_SIZE,), fill_value=-1) + GAMMA * torch.ones(
        size=(BATCH_SIZE,)
    )
    targets[0] = 0.0  # first is terminal
    critic_loss = torch.nn.functional.mse_loss(values, targets)
    advantage = targets - values

    # actor_loss
    actor_loss = (-log_probs * advantage).mean()

    assert torch.allclose(loss, critic_loss + actor_loss)
