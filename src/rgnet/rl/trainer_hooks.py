import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict

import torch
from tensordict import NestedKey
from torchrl.trainers import Trainer, TrainerHookBase

from rgnet.rl import Agent
from rgnet.rl.envs.planning_env import PlanningEnvironment


class EarlyStoppingTrainerHook(TrainerHookBase, metaclass=ABCMeta):

    def __init__(self):
        self.trainer = None

    def state_dict(self) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def should_stop(self) -> bool:
        pass

    def __call__(self, *args, **kwargs):
        if self.should_stop():
            logging.info("Early stopping condition met. Stopping training.")
            self.trainer.total_frames = 1

    def register(self, trainer: Trainer, name: str):
        trainer.register_op("post_steps", self)
        self.trainer = trainer


class ValueFunctionConverged(EarlyStoppingTrainerHook):

    def __init__(
        self,
        value_operator,
        reset_func,
        optimal_values_lookup,
        atol=0.1,
        state_value_key=Agent.default_keys.state_value,
    ):
        super().__init__()
        self.value_operator = value_operator
        self.reset_func = reset_func
        self.optimal_values_lookup = optimal_values_lookup
        self.atol = atol
        self.state_value_key = state_value_key
        self.state_value_history = []

    def should_stop(self) -> bool:
        td = self.reset_func()
        with torch.no_grad():
            self.value_operator.eval()
            predicted_values: torch.Tensor = (
                self.value_operator(td).get(self.state_value_key).squeeze(-1)
            )
            self.state_value_history.append(predicted_values.detach().cpu())
            self.value_operator.train()
            true_values = torch.stack(
                [
                    self.optimal_values_lookup[s]
                    for s in td[PlanningEnvironment.default_keys.state]
                ]
            )
            return torch.allclose(true_values, predicted_values, atol=self.atol)


class ConsecutiveStopping(EarlyStoppingTrainerHook):

    def __init__(self, times: int, stopping_module):
        super().__init__()
        self.stopping_module = stopping_module
        self.times = times
        self.counter = 0

    def should_stop(self, *args, **kwargs):
        if self.stopping_module.should_stop():
            self.counter += 1
            return self.counter >= self.times
        else:
            self.counter = 0
            return False


class LoggingHook(TrainerHookBase):

    def state_dict(self) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def __init__(self, probs_key: NestedKey, action_key: NestedKey):
        super().__init__()
        self.action_key = action_key
        self.probs_key = probs_key
        self.probs_history = []
        self.values_history = []
        self.done_samples = 0
        self.selected_actions = []

    def __call__(self, batch):
        dones: torch.Tensor = batch[("next", "done")]
        self.done_samples += dones.count_nonzero().item()
        if self.probs_key in batch:
            self.probs_history.append(batch[self.probs_key])
        self.selected_actions.append(batch[self.action_key])

    def register(self, trainer: Trainer, name: str):
        trainer.register_op("post_steps_log", self)
