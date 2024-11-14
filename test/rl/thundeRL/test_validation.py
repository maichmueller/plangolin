from test.fixtures import expanded_state_space_env, medium_blocks  # noqa: F401

import lightning
import mockito
import pytest
import torch
from torchrl.modules import ValueOperator

from rgnet.rl.thundeRL.validation import CriticValidation


@pytest.mark.parametrize(
    "expanded_state_space_env", [["medium_blocks", 25]], indirect=True
)
def test_critic_validation(expanded_state_space_env):
    """
    Tests the critic validation callback.
    Go over whole state space in 5 batches, each 25 elements.
    The value operator is mocked and produces estimates one higher than optimal.
    The optimal values are just 0,...,124, but the indices of states are reversed.
    Therefore, the callback has to sort the predictions based on the idx_in_space.
    The expected loss is 1.0 = torch.nn.functional.mse([0,...,124],[1,...,125])
    """
    env = expanded_state_space_env
    batch_size = 25
    idx_in_space = torch.arange(start=124, end=-1, step=-1, dtype=torch.long)
    overestimated_values = torch.arange(start=125, end=0, step=-1, dtype=torch.float)

    class OverestimatingValueOperator(torch.nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.batch_index = 0

        def forward(self, states):
            values = overestimated_values[
                batch_size * self.batch_index : batch_size * (self.batch_index + 1)
            ]
            self.batch_index += 1
            return values.unsqueeze(dim=-1)

    optimal_values_dict = {0: torch.arange(0, 125, dtype=torch.float)}

    value_operator = ValueOperator(
        in_keys=["state"], module=OverestimatingValueOperator()
    )
    critic_validation = CriticValidation(
        discounted_optimal_values=optimal_values_dict, value_operator=value_operator
    )
    for batch_index in range(5):
        rollout = env.rollout(max_steps=1)
        rollout["idx_in_space"] = idx_in_space[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ].unsqueeze(dim=-1)
        critic_validation(rollout)

    d = dict()

    def log_mock(name, key, on_epoch: bool = True):
        d[name] = key

    pl_mock = mockito.mock({"log": log_mock}, spec=lightning.LightningModule)
    critic_validation.on_validation_epoch_end(
        mockito.mock(spec=lightning.Trainer), pl_mock
    )
    values = list(d.values())
    assert len(values) == 1
    loss = values[0]
    assert loss == pytest.approx(1.0)
