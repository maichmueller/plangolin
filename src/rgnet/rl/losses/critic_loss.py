from dataclasses import dataclass
from typing import Literal, Optional

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator
from torchrl.objectives import LossModule, ValueEstimators, distance_loss
from torchrl.objectives.utils import _reduce
from torchrl.objectives.value import GAE, TD0Estimator, TD1Estimator, TDLambdaEstimator

from rgnet.rl.losses.all_actions_estimator import AllActionsValueEstimator


class CriticLoss(LossModule):
    @dataclass(frozen=True)
    class _AcceptedKeys:
        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey = "log_probs"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    tensor_keys: _AcceptedKeys
    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        critic_network: ValueOperator,
        reduction: Optional[str] = None,
        loss_critic_type: str = "l2",
        clone_tensordict: bool = True,
        keys: _AcceptedKeys = default_keys,
    ):
        super().__init__()
        self.critic_network = critic_network
        self.reduction: str = reduction or "mean"
        self.loss_critic_type: str = loss_critic_type
        self.clone_tensordict: bool = clone_tensordict
        self._tensor_keys: CriticLoss._AcceptedKeys = keys

    @property
    def loss_components(self):
        return ["loss_critic"]

    def _loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        # value_target should have been computed with the advantage in forward
        target_return = tensordict.get(self.tensor_keys.value_target)
        # We have to always recompute as the ValueEstimator might use a copy of the actual parameter
        # which means the result might still have gradients (just for non-nonsensical parameter)
        # TODO get the value estimator to pass through gradients for the value_net in order to avoid second call
        tensordict_select = tensordict.select(
            *self.critic_network.in_keys, strict=False
        )
        state_value = self.critic_network(tensordict_select).get(self.tensor_keys.value)

        return distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )

    def forward(self, tensordict_in: TensorDictBase):
        if self.clone_tensordict:
            tensordict = tensordict_in.clone(False)
        else:
            tensordict = tensordict_in
        self.value_estimator(tensordict)
        loss_critic = self._loss_critic(tensordict)
        batch_less_loss = _reduce(loss_critic, self.reduction)
        return TensorDict({self.loss_components[0]: batch_less_loss}, batch_size=[])

    def make_value_estimator(
        self,
        value_type: ValueEstimators | Literal["AllActionsValueEstimator"] = None,
        optimal_targets: Optional[TensorDictModule] = None,
        **hyperparams,
    ):
        """Taken from ReinforceLoss.make_value_estimator."""
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        value_network_for_target_values = self.critic_network
        if optimal_targets:
            value_network_for_target_values = optimal_targets
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=value_network_for_target_values, **hyperparams
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=value_network_for_target_values, **hyperparams
            )
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(
                value_network=value_network_for_target_values, **hyperparams
            )
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=value_network_for_target_values, **hyperparams
            )
        elif value_type == "AllActionsValueEstimator":
            self._value_estimator = AllActionsValueEstimator(
                value_network=value_network_for_target_values, **hyperparams
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value": self.tensor_keys.value,
            "value_target": self.tensor_keys.value_target,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
            "sample_log_prob": self.tensor_keys.sample_log_prob,
        }
        self._value_estimator.set_keys(**tensor_keys)
