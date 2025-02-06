from typing import Tuple

import mockito
import pytest
import torch
from tensordict import LazyStackedTensorDict, NonTensorData, NonTensorStack, TensorDict
from torchrl.modules import ValueOperator

from rgnet.rl.agents import ActorCritic
from rgnet.rl.losses import AllActionsLoss, AllActionsValueEstimator


class TestAllActionsLoss:
    """
    Covered test scenarios:
        - End to end usage with test for specific results
            - Mock AllActionsValueEstimator
            - Only 1 batch_entry
            - No time dimension
    Not covered:
        - Gradients computed correctly -> check that advantage is detached
        - batch_size > 1
        - Multiple time steps
        - Error handling
    """

    actor_keys = ActorCritic.default_keys
    estimator_keys = AllActionsValueEstimator.default_keys
    device = torch.device("cpu")

    def _stacked_grad_tensor(self, list_of_floats: list[float]):
        return NonTensorStack(
            NonTensorData(
                torch.tensor(
                    list_of_floats,
                    requires_grad=True,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        )

    @pytest.fixture
    def critic_mock(self):
        keys = [TestAllActionsLoss.estimator_keys.value]
        critic = ValueOperator(module=torch.nn.Identity(), in_keys=keys, out_keys=keys)
        mockito.spy2(critic.forward)
        return critic

    def data(self) -> Tuple[TensorDict, torch.Tensor, torch.Tensor]:
        r"""
        batch_size = 1, num_successors = [2], gamma=0.999

        One state with two successors
        s0 -> s1, s0 -> s2
        v(s1) = -1, v(s0) = -2, v(s2) = -3
        \pi(s_1|s_0) = 0.4, \pi(s_2 |s_0) = 0.4

        targets:
        target(s_1) = -1 + 0.999 * -1 = -1.0
        target(s_2) = -1 + 0.999 * -3 = -2.998

        expected_target:
        target(s_0) = 0.6 * (-2.998) + 0.4 * (-1) = -2.1988

        successor_advantage:
        A(s_1) = -1 - (-2.1988) = 1.1988
        A(s_2) = -2.998 - (-2.1988) = -0.7992

        :return: Tensordict with state_value, value_target, successor_advantage, probs
        """
        td = TensorDict(
            {
                self.actor_keys.probs: self._stacked_grad_tensor([0.4, 0.6]),
                self.estimator_keys.successor_advantage: self._stacked_grad_tensor(
                    [1.1988, -0.7992]
                ),
                self.estimator_keys.value_target: torch.tensor(
                    -2.1988, device=self.device
                ).view(1, 1, 1),
                self.estimator_keys.value: torch.tensor(-2, device=self.device).view(
                    1, 1, 1
                ),
            },
            batch_size=(1,),
        )
        td_with_time = LazyStackedTensorDict.maybe_dense_stack([td], 1)
        td_with_time.refine_names(..., "time")

        expected_actor_loss = torch.tensor(
            -0.4 * 1.1988 - 0.6 * -0.7992, device=self.device
        )
        expected_critic_loss = torch.nn.functional.mse_loss(
            torch.tensor(-2.1988), torch.tensor(-2)
        )
        return td_with_time, expected_actor_loss, expected_critic_loss

    def test_all_actions_loss_forward(self, critic_mock):
        test_td: TensorDict
        expected_actor_loss: torch.Tensor
        expected_critic_loss: torch.Tensor
        test_td, expected_actor_loss, expected_critic_loss = self.data()

        all_actions_estimator_mock = mockito.mock(
            {
                "__call__": lambda tensordict: tensordict.update(test_td, inplace=True),
                "compute_successor_advantages": True,
                "tensor_keys": TestAllActionsLoss.estimator_keys,
            },
            spec=AllActionsValueEstimator,
        )

        loss = AllActionsLoss(
            critic_network=critic_mock,
            loss_critic_type="l2",
            reduction="mean",
            clone_tensordict=True,
        )
        loss._value_estimator = all_actions_estimator_mock
        assert loss.loss_components == ["loss_critic", "loss_actor"]

        loss_output = loss(
            test_td.select(self.actor_keys.probs, self.estimator_keys.value)
        )

        assert set(loss_output.keys()) == set(loss.loss_components)

        loss_actor = loss_output["loss_actor"]
        loss_critic = loss_output["loss_critic"]
        assert torch.allclose(loss_actor, expected_actor_loss, atol=1e-4)
        assert torch.allclose(loss_critic, expected_critic_loss, atol=1e-4)
