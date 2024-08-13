from dataclasses import dataclass
from typing import Optional

from tensordict import NestedKey, TensorDict, TensorDictBase
from torchrl.modules import ValueOperator
from torchrl.objectives.utils import ValueEstimators, _reduce

from rgnet.rl.losses.critic_loss import CriticLoss


class ActorCriticLoss(CriticLoss):

    default_value_estimator: ValueEstimators = ValueEstimators.TD0

    @dataclass(frozen=True)
    class _AcceptedKeys(CriticLoss._AcceptedKeys):
        sample_log_prob: NestedKey = "log_probs"

    default_keys = _AcceptedKeys()
    tensor_keys: _AcceptedKeys

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
        super().__init__(
            critic_network=critic_network,
            reduction=reduction,
            loss_critic_type=loss_critic_type,
            clone_tensordict=clone_tensordict,
        )
        self.log_prob_clip = log_prob_clip_value
        self._tensor_keys = keys

    @property
    def loss_components(self):
        return ["loss_critic", "loss_actor"]

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
