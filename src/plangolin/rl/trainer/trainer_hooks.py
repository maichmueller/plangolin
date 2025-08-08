from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List

import torch
from tensordict import NestedKey, TensorDict
from torchrl.modules import ValueOperator
from torchrl.trainers import Trainer, TrainerHookBase

import xmimir as xmi
from plangolin.logging_setup import get_logger
from plangolin.rl.agents import ActorCritic
from plangolin.rl.envs.planning_env import PlanningEnvironment


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
            get_logger(__name__).info(
                "Early stopping condition met. Stopping training."
            )
            self.trainer.total_frames = 1

    def register(self, trainer: Trainer, name: str):
        trainer.register_op("post_steps", self)
        self.trainer = trainer


class ValueFunctionConverged(EarlyStoppingTrainerHook):
    def __init__(
        self,
        value_operator,
        reset_func,
        optimal_values_lookup: Dict[xmi.State, float],
        atol=0.1,
        state_value_key=ActorCritic.default_keys.state_value,
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
            true_values = torch.tensor(
                [
                    self.optimal_values_lookup[s]
                    for s in td[PlanningEnvironment.default_keys.state]
                ],
                device=predicted_values.device,
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

    def __init__(self, logging_keys: List[NestedKey]):
        super().__init__()
        self.logging_keys = logging_keys
        self.done_samples = []
        self.logging_dict = {key: [] for key in logging_keys}

    def __call__(self, batch):
        dones: torch.Tensor = batch[("next", "done")]
        self.done_samples.append(dones.count_nonzero().item())
        for key in self.logging_keys:
            if key in batch:
                self.logging_dict[key].append(batch[key])

    def register(self, trainer: Trainer, name: str):
        trainer.register_op("post_steps_log", self)


class ValueFunctionLoggingHook(TrainerHookBase):
    def __init__(
        self,
        td_generator: Callable[[], TensorDict],
        value_operator: ValueOperator,
        interval: int,
    ):
        super().__init__()
        self.value_operator = value_operator
        self.td_generator = td_generator
        self.prediction_history: List[torch.Tensor] = []
        self.interval = interval
        self.counter = 0

    def state_dict(self) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def __call__(self):
        if self.counter % self.interval == 0:
            td = self.td_generator()
            self.prediction_history.append(self.value_operator(td))
        self.counter += 1

    def register(self, trainer: Trainer, name: str):
        trainer.register_op("post_steps_hook", self)
