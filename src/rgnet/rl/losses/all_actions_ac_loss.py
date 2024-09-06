from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator
from torchrl.objectives import ValueEstimators
from torchrl.objectives.utils import _reduce

from rgnet.rl import ActorCritic
from rgnet.rl.losses import CriticLoss
from rgnet.rl.losses.all_actions_estimator import AllActionsValueEstimator


def _loss_actor(
    advantages: torch.Tensor,
    probs: torch.Tensor,
):
    if (*probs.shape, 1) == advantages.shape:
        probs = probs.unsqueeze(dim=-1)

    return (-probs * advantages.detach()).sum()


class AllActionsLoss(CriticLoss):
    @dataclass(frozen=True)
    class _AcceptedKeys(CriticLoss._AcceptedKeys):
        probs: NestedKey = ActorCritic.default_keys.probs

    default_keys = _AcceptedKeys()
    tensor_keys: _AcceptedKeys
    # AllActionsLoss only works with AllActionsValueEstimator
    # We need especially need the advantage values for each successor state from it.
    value_estimator: AllActionsValueEstimator
    default_value_estimator = "AllActionsValueEstimator"

    def __init__(
        self,
        critic_network: ValueOperator,
        reduction: Optional[str] = None,
        loss_critic_type: str = "l2",
        clone_tensordict: bool = True,
        keys: _AcceptedKeys = default_keys,
    ):
        super().__init__(
            critic_network, reduction, loss_critic_type, clone_tensordict, keys
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        if self.clone_tensordict:
            tensordict = tensordict.clone(False)
        # Shape is batch x time x 1
        individual_advantages: torch.Tensor = tensordict.get(
            self.value_estimator.tensor_keys.individual_advantage, None
        )
        if individual_advantages is None:
            self.value_estimator(tensordict)
            individual_advantages = tensordict.get(
                self.value_estimator.tensor_keys.individual_advantage
            )
        individual_advantages: List[List[torch.Tensor]] = individual_advantages.tolist()
        # Shape is batch x time x num_successors
        probs: List[List[torch.Tensor]] = tensordict.get(
            self.tensor_keys.probs
        ).tolist()
        # Values for each successor state over the batch over each time step
        # Shape is batch x time x num_successors x 1
        batch_size = len(individual_advantages)
        loss_actor: List[torch.Tensor] = []

        for batch_entry in range(batch_size):
            loss_over_time: List[torch.Tensor] = []
            for time in range(len(individual_advantages[batch_entry])):
                loss_over_time.append(
                    _loss_actor(
                        individual_advantages[batch_entry][time],
                        probs[batch_entry][time],
                    )
                )
            loss_actor.append(torch.stack(loss_over_time))

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
        hyperparams["compute_individual_advantages"] = True
        return super().make_value_estimator(value_type, optimal_targets, **hyperparams)
