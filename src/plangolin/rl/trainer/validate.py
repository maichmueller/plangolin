from typing import Callable, Dict, List, Set

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type

import xmimir as xmi
from plangolin.rl.envs.planning_env import PlanningEnvironment
from plangolin.utils.misc import as_non_tensor_stack


class SupervisedValueLoss(torch.nn.Module):
    def __init__(
        self,
        optimal_values: Dict[xmi.XStateSpace, torch.Tensor],
        value_module: Callable[[List[xmi.XState]], torch.Tensor],
        device: torch.device,
        loss_function=torch.nn.functional.mse_loss,
        log_name: str = "value_loss",
    ):
        super().__init__()
        self.optimal_values = optimal_values
        self.module = value_module
        self.device = device
        self.loss_function = loss_function
        self.log_name: str = log_name

    def forward(self):
        losses = dict()
        for space, values in self.optimal_values.items():
            prediction = self.module(list(space.states_iter())).squeeze(dim=-1)
            values = values.to(prediction.device)
            loss = self.loss_function(prediction, values)
            losses[f"{self.log_name}/{space.problem.name}"] = loss
        return losses


class PolicyQuality(torch.nn.Module):
    def __init__(
        self,
        spaces: List[xmi.XStateSpace],
        policy: TensorDictModule,
        keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
        log_name: str = "policy_quality",
    ) -> None:
        super().__init__()
        self.policy = policy
        self.test_data = {
            space: TensorDict(
                {
                    keys.state: as_non_tensor_stack(space),
                    keys.transitions: as_non_tensor_stack(
                        [list(space.forward_transitions(s)) for s in space]
                    ),
                },
                batch_size=torch.Size((len(space),)),
            )
            for space in spaces
        }
        self.optimal_transitions: Dict[
            xmi.XStateSpace, Dict[xmi.XState, Set[xmi.XTransition]]
        ] = {
            space: {s: PolicyQuality.optimal_actions(space, s) for s in space}
            for space in spaces
        }
        self.keys = keys
        self.log_name = log_name

    @staticmethod
    def optimal_actions(
        space: xmi.XStateSpace, state: xmi.XState
    ) -> Set[xmi.XTransition]:
        best_distance = min(
            space.goal_distance(t.target) for t in space.forward_transitions(state)
        )
        return set(
            t
            for t in space.forward_transitions(state)
            if space.goal_distance(t.target) == best_distance
        )

    @torch.no_grad()
    def forward(self):
        # TODO use cross entropy loss between all optimal actions and policy probs
        losses = dict()
        with set_exploration_type(ExplorationType.MODE):
            self.policy.eval()

            for space in self.test_data.keys():
                num_correct_actions = self.num_correct_actions(space)
                policy_precision: float = num_correct_actions / float(len(space))
                losses[f"{self.log_name}/{space.problem.name}"] = policy_precision

            self.policy.train()
            return losses

    def num_correct_actions(self, space: xmi.XStateSpace):
        actions: List[xmi.XTransition] = self.policy(self.test_data[space])[
            self.keys.action
        ]
        optimal_actions: Dict[xmi.XState, Set[xmi.XTransition]] = (
            self.optimal_transitions[space]
        )
        return sum(action in optimal_actions[action.source] for action in actions)
