from dataclasses import dataclass
from typing import Optional

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from torchrl.modules import ValueOperator
from torchrl.objectives import LossModule
from torchrl.objectives.utils import (
    ValueEstimators,
    _reduce,
    default_value_kwargs,
    distance_loss,
)
from torchrl.objectives.value import GAE, TD0Estimator, TD1Estimator, TDLambdaEstimator


class ActorCriticLoss(LossModule):

    default_value_estimator: ValueEstimators = ValueEstimators.TD0

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

    def __init__(
        self,
        critic_network: ValueOperator,
        reduction: Optional[str] = None,
        loss_critic_type: str = "l2",
        log_prob_clip_value: Optional[float] = None,
        clone_tensordict: bool = True,
        keys: _AcceptedKeys = default_keys,
    ):
        """
        ActorCriticLoss is nearly identical to a simplified version of ReinforceLoss.
        The core difference is that we do not require the Actor (ProbabilisticActor) and
        instead assume that the log_probs are already present in the tensordict.
        This loss does neither work with functional modules nor with clipped losses.
        Specify the ValueEstimator type using the make_value_estimator method.
            Defaults to TD0Estimator

        :param critic_network: Module that estimates the state-value.
        :param reduction: reduction method used to aggregate the loss across the batch dim.
        :param loss_critic_type: The loss function used to compute the critic loss.
            Defaults to l2 (mse).
        :param clone_tensordict: If False intermediate results (like advantage etc) will be written
            into the input tensordict.
            Defaults to True.
        """
        super().__init__()
        self.loss_critic_type: str = loss_critic_type
        self.critic_network: ValueOperator = critic_network
        self.reduction: str = reduction or "mean"
        self.log_prob_clip = log_prob_clip_value
        self.clone_tensordict: bool = clone_tensordict
        self._tensor_keys = keys

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """
        Compute the loss_critic and loss_actor given the rollout in tensordict.
        :param tensordict: The tensordict contain a rollout.
          Required keys are:
            - The in-keys of the critic_network
            - self.tensor_keys.sample_log_prob
            - The keys for the ValueEstimator
          Optional-keys:
            - If self.tensor_keys.advantage is present the ValueEstimator is not called.
                Note if advantage is present value_target should be present too.
        :return: A new tensordict containing both actor and critic loss. Note that
            this tensordict will have no batch-dimension as the loss is aggregated.
        """
        if self.clone_tensordict:
            tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(tensordict)
            advantage = tensordict.get(self.tensor_keys.advantage)
        # We assume that the log_probs are already present in the tensordict.
        # Use return_log_prob=True in the ProbabilisticTensorDictModule.
        log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if log_prob.shape == advantage.shape[:-1]:
            log_prob = log_prob.unsqueeze(-1)
        # The advantage could have gradients from the critic_network which should not
        # influence the actor-loss.
        if self.log_prob_clip:
            log_prob = log_prob.clamp(-self.log_prob_clip, self.log_prob_clip)
        loss_actor = -log_prob * advantage.detach()
        td_out = TensorDict({"loss_actor": loss_actor}, batch_size=[])

        loss_value = self._loss_critic(tensordict)
        td_out.set("loss_critic", loss_value)
        td_out = td_out.named_apply(
            lambda name, value: (
                _reduce(value, reduction=self.reduction).squeeze(-1)
                if name.startswith("loss_")
                else value
            ),
            batch_size=[],
        )

        return td_out

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

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        """Taken from ReinforceLoss.make_value_estimator."""
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(value_network=self.critic_network, **hp)
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=self.critic_network, **hp
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
