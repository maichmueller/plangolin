from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator
from torchrl.objectives import ValueEstimators
from torchrl.objectives.utils import _reduce

from rgnet.rl import ActorCritic
from rgnet.rl.losses import AllActionsValueEstimator, CriticLoss

AllActionsStubType = Optional[
    Callable[[ValueOperator], AllActionsValueEstimator] | AllActionsValueEstimator
]


class AllActionsLoss(CriticLoss):
    @dataclass(frozen=True)
    class _AcceptedKeys(CriticLoss._AcceptedKeys):
        probs: NestedKey = ActorCritic.default_keys.probs

    default_keys = _AcceptedKeys()
    tensor_keys: _AcceptedKeys
    # AllActionsLoss only works with AllActionsValueEstimator
    # We especially need the advantage values for each successor state from it.
    value_estimator: AllActionsValueEstimator
    default_value_estimator = "AllActionsValueEstimator"

    def __init__(
        self,
        critic_network: ValueOperator,
        reduction: Optional[Literal["mean", "sum", "max"]] = None,
        loss_critic_type: str = "l2",
        value_estimator: AllActionsStubType = None,
        clone_tensordict: bool = True,
        keys: _AcceptedKeys = default_keys,
    ):
        super().__init__(
            critic_network,
            reduction,
            loss_critic_type,
            value_estimator=value_estimator,
            clone_tensordict=clone_tensordict,
            keys=keys,
        )
        assert isinstance(self.value_estimator, AllActionsValueEstimator), (
            "AllActionsLoss only works with AllActionsValueEstimator,"
            " because the advantage of all available actions is required."
        )

    @property
    def loss_components(self):
        critic_loss = super().loss_components
        critic_loss.append("loss_actor")
        return critic_loss

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        if self.clone_tensordict:
            tensordict = tensordict.clone(False)

        # We require the advantage for each outgoing transition.
        assert self.value_estimator.compute_successor_advantages
        # noinspection PyTypeChecker
        opt_successor_advantages: Optional = tensordict.get(
            self.value_estimator.tensor_keys.successor_advantage, None
        )
        if opt_successor_advantages is None:
            self.value_estimator(tensordict)
            opt_successor_advantages = tensordict.get(
                self.value_estimator.tensor_keys.successor_advantage
            )
        assert opt_successor_advantages is not None
        # Shape for both is batch x time x num_successors
        # we use .get() which returns `NonTensorStack`
        # [batch_size, time, num_successor, ]
        successor_advantages: List[List[torch.Tensor]] = (
            opt_successor_advantages.tolist()
        )
        # pi(s' \mid s) for s' in N(s) as computed by the agent.
        # [batch_size, time, num_successor, ]
        transition_probabilities: List[List[torch.Tensor]] = tensordict[
            self.tensor_keys.probs
        ]
        # Values for each successor state over the batch over each time step
        # Shape is [batch_size, time,]
        loss_actor: List[torch.Tensor] = []

        def _loss_actor(tpl: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            # advantages and transition probabilities for one specific state and time step
            advantages, transitions_probability = tpl
            if (*transitions_probability.shape, 1) == advantages.shape:
                transitions_probability = transitions_probability.unsqueeze(dim=-1)

            return (-transitions_probability * advantages.detach()).sum()

        for batch_advantages, batch_probs in zip(
            successor_advantages, transition_probabilities
        ):
            loss_actor.append(
                torch.stack(tuple(map(_loss_actor, zip(batch_advantages, batch_probs))))
            )

        td_out = TensorDict({"loss_actor": torch.stack(loss_actor)}, batch_size=[])

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

    def make_value_estimator(
        self,
        value_type: ValueEstimators | Literal["AllActionsValueEstimator"] = None,
        optimal_targets: Optional[TensorDictModule] = None,
        **hyperparams,
    ):
        if value_type != "AllActionsValueEstimator":
            raise ValueError("AllActionsLoss only works with AllActionsValueEstimator.")
        hyperparams["compute_successor_advantages"] = True
        return super().make_value_estimator(value_type, optimal_targets, **hyperparams)
