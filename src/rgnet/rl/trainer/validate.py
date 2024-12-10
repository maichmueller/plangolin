from typing import Callable, Dict, List, Set

import pymimir as mi
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type

from rgnet.rl.envs.planning_env import PlanningEnvironment
from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack


class SupervisedValueLoss(torch.nn.Module):
    def __init__(
        self,
        optimal_values: Dict[mi.StateSpace, torch.Tensor],
        value_module: Callable[[List[mi.State]], torch.Tensor],
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
            prediction = self.module(space.get_states()).squeeze(dim=-1)
            values = values.to(prediction.device)
            loss = self.loss_function(prediction, values)
            losses[f"{self.log_name}/{space.problem.name}"] = loss
        return losses


class PolicyQuality(torch.nn.Module):
    def __init__(
        self,
        spaces: List[mi.StateSpace],
        policy: TensorDictModule,
        keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
        log_name: str = "policy_quality",
    ) -> None:
        super().__init__()
        self.policy = policy
        self.test_data = {
            space: TensorDict(
                {
                    keys.state: as_non_tensor_stack(space.get_states()),
                    keys.transitions: as_non_tensor_stack(
                        [space.get_forward_transitions(s) for s in space.get_states()]
                    ),
                },
                batch_size=torch.Size((space.num_states(),)),
            )
            for space in spaces
        }
        self.optimal_transitions: Dict[
            mi.StateSpace, Dict[mi.State, Set[mi.Transition]]
        ] = {
            space: {
                s: PolicyQuality.optimal_actions(space, s) for s in space.get_states()
            }
            for space in spaces
        }
        self.keys = keys
        self.log_name = log_name

    @staticmethod
    def optimal_actions(space: mi.StateSpace, state: mi.State) -> Set[mi.Transition]:
        best_distance = min(
            space.get_distance_to_goal_state(t.target)
            for t in space.get_forward_transitions(state)
        )
        return set(
            t
            for t in space.get_forward_transitions(state)
            if space.get_distance_to_goal_state(t.target) == best_distance
        )

    @torch.no_grad()
    def forward(self):
        # TODO use cross entropy loss between all optimal actions and policy probs
        losses = dict()
        with set_exploration_type(ExplorationType.MODE):
            self.policy.eval()

            for space in self.test_data.keys():
                num_correct_actions = self.num_correct_actions(space)
                policy_precision: float = num_correct_actions / float(
                    space.num_states()
                )
                losses[f"{self.log_name}/{space.problem.name}"] = policy_precision

            self.policy.train()
            return losses

    def num_correct_actions(self, space: mi.StateSpace):
        actions: List[mi.Transition] = self.policy(self.test_data[space])[
            self.keys.action
        ]
        optimal_actions: Dict[mi.State, Set[mi.Transition]] = self.optimal_transitions[
            space
        ]
        return sum(action in optimal_actions[action.source] for action in actions)
