import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import torch
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
        optimal_values,
        atol=0.1,
        state_key=Agent.default_keys.state_value,
    ):
        super().__init__()
        self.value_operator = value_operator
        self.reset_func = reset_func
        self.optimal_values = optimal_values
        self.atol = atol
        self.state_key = state_key
        self.state_value_history = []

    def should_stop(self) -> bool:
        td = self.reset_func()
        with torch.no_grad():
            self.value_operator.eval()
            predicted_values: torch.Tensor = (
                self.value_operator(td).get(self.state_key).squeeze(-1)
            )
            self.state_value_history.append(predicted_values.detach().cpu())
            self.value_operator.train()
            return torch.allclose(self.optimal_values, predicted_values, atol=self.atol)


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
