import copy
import itertools

import mockito
import pytest
import torch.nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.modules import ValueOperator
from torchrl.objectives import ValueEstimators

from rgnet.rl import SimpleLoss


@pytest.fixture
def rollout_not_done(request):

    batch_size: int = request.param[0]
    rollout_length: int = request.param[1]
    hidden_size = 3

    return TensorDict(
        {
            "observation": torch.rand((batch_size, rollout_length, hidden_size)),
            "next": TensorDict(
                {
                    "observation": torch.rand(
                        (batch_size, rollout_length, hidden_size)
                    ),
                    "reward": torch.ones((batch_size, rollout_length, 1)),
                    "done": torch.zeros(
                        (batch_size, rollout_length, 1), dtype=torch.bool
                    ),
                    "terminated": torch.zeros(
                        (batch_size, rollout_length, 1), dtype=torch.bool
                    ),
                },
                batch_size=batch_size,
            ),
        },
        batch_size=torch.Size([batch_size]),
    )


@pytest.fixture
def critic_mock(hidden_size=3):
    linear = nn.Linear(hidden_size, 1)
    vo = ValueOperator(linear)
    mockito.spy2(vo.forward)
    return vo


@pytest.fixture
def actor_mock(hidden_size=3):
    class M(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, obs):
            c = Categorical(probs=self.linear(obs).softmax(dim=-1))
            s = c.sample()
            return c.log_prob(s)

    tdm = TensorDictModule(
        module=M(),
        in_keys=["observation"],
        out_keys=[SimpleLoss.default_keys.sample_log_prob],
    )
    mockito.spy2(tdm.forward)
    return tdm


@pytest.mark.parametrize("rollout_not_done", [[1, 2], [1, 5]], indirect=True)
def test_forward(critic_mock, actor_mock, rollout_not_done):
    gamma = 0.9
    loss = SimpleLoss(critic_mock, reduction="mean")
    loss.make_value_estimator(ValueEstimators.TD0, gamma=gamma, shifted=True)

    optim = torch.optim.SGD(
        itertools.chain(critic_mock.parameters(), actor_mock.parameters())
    )

    actor_mock(rollout_not_done)  # add log_probs with gradients

    expected_keys = copy.deepcopy(rollout_not_done.sorted_keys)
    loss_out: TensorDict = loss(rollout_not_done)

    # Default behavior will not write intermediate results into the input tensordict.
    assert rollout_not_done.sorted_keys == expected_keys
    assert loss_out.batch_size == torch.Size([])
    assert loss_out.sorted_keys == ["loss_actor", "loss_critic"]
    loss_actor = loss_out["loss_actor"]
    loss_critic = loss_out["loss_critic"]

    assert loss_actor.requires_grad
    assert loss_critic.requires_grad

    # Critic network is called once for the advantage (no gradients) and once for the state_value (gradients).
    mockito.verify(critic_mock, times=2).forward(...)
    mockito.verify(actor_mock, times=1).forward(...)

    # Verify the losses
    with torch.no_grad():
        prediction = critic_mock(rollout_not_done.select(*critic_mock.in_keys)).get(
            SimpleLoss.default_keys.value
        )
        loss.value_estimator(rollout_not_done)
        expected_value_target = rollout_not_done[SimpleLoss.default_keys.value_target]
        expected_advantage = rollout_not_done[SimpleLoss.default_keys.advantage]
        expected_loss_critic = torch.nn.functional.mse_loss(
            prediction, expected_value_target, reduction="mean"
        )
        assert torch.allclose(loss_out["loss_critic"], expected_loss_critic)
        log_probs = rollout_not_done[SimpleLoss.default_keys.sample_log_prob].unsqueeze(
            -1
        )
        expected_loss_actor = (-log_probs * expected_advantage.detach()).mean()

        assert torch.allclose(loss_out["loss_actor"], expected_loss_actor)

    loss_actor.backward()
    optim.step()

    assert all(param.grad is None for param in critic_mock.parameters())

    assert not all(param.grad is None for param in actor_mock.parameters())

    optim.zero_grad()
    loss_critic.backward()
    optim.step()

    assert not all(param.grad is None for param in critic_mock.parameters())
