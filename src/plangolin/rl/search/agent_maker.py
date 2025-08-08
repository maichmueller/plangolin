from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Iterable, Optional

import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase, TransformedEnv

from plangolin.encoding import GraphEncoderBase
from plangolin.rl.agents import LookupPolicyActor
from plangolin.rl.data_layout import InputData, OutputData
from plangolin.rl.envs import PlanningEnvironment
from xmimir import XProblem


class AgentMaker(ABC):
    def __init__(
        self,
        module: torch.nn.Module | TensorDictModule,
        *,
        device: str | torch.device,
        **kwargs,
    ):
        self.module = module
        self.device = device

    @abstractmethod
    def agent(
        self,
        checkpoint_path: Path,
        instance: XProblem,
        epoch: int = None,
        **kwargs,
    ) -> TensorDictModule:
        """
        Return the actor module from the model and given checkpoint that can be used to interact with the environment.

        The actor may be instance specific, but in general is independent of the instance (e.g. a general policy).
        """
        pass

    @abstractmethod
    def transformed_env(
        self, base_env: PlanningEnvironment
    ) -> PlanningEnvironment | TransformedEnv:
        """
        Return a transformed environment that provides the necessary setup to work with the agent.
        This is used to run rollouts on the environment.
        """
        pass

    @property
    def encoder(self) -> GraphEncoderBase | None:
        """
        Get the encoder for the agent maker. This is used to encode states into a graph representation.
        May be `None` if no encoder is needed. Should also be set to None so that the maker can be sent across process
        boundaries.
        """
        return None

    @encoder.setter
    def encoder(self, encoder: GraphEncoderBase):
        pass


class PreComputedAgentMaker(AgentMaker):
    def __init__(
        self,
        in_data: InputData,
        out_data: OutputData,
        *,
        device: str | torch.device,
        env_keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
        state_space_options: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            module=torch.nn.Identity(),  # irrelevant for pre-computed models
            device=device,
            **kwargs,
        )
        self.in_data = in_data
        self.out_data = out_data
        self.state_space_options = state_space_options or {}
        self.env_keys = env_keys

    def agent(
        self, instance: XProblem, checkpoint_path: Path, epoch: int = None, **kwargs
    ) -> TensorDictModule:
        epoch_list = self.stored_probs(instance)
        if len(epoch_list) == 0:
            raise ValueError(f"No probabilities were saved for problem {instance}.")
        probs_list: list[torch.Tensor] = (
            epoch_list[-1][1]  # get latest and take list of tensors
            if epoch is None
            else next(probs for (ep, probs) in epoch_list if ep == epoch)
        )
        if (
            space := self.in_data.get_or_load_space(
                instance,
            )
        ) is not None:
            idx_of_state = space
        else:
            idx_of_state = "idx_in_space"
        return LookupPolicyActor(
            probs_list, instance, env_keys=self.env_keys, idx_of_state=idx_of_state
        )

    def transformed_env(self, base_env: PlanningEnvironment) -> EnvBase:
        return base_env

    @cache
    def find_problem_by_name(self, name: str):
        problem_sources = [
            self.in_data.problems,
            self.in_data.validation_problems,
            self.in_data.test_problems,
        ]
        try:
            return next(
                filter(
                    lambda x: x is not None,
                    (
                        self.problem_matching_name(name, source)
                        for source in problem_sources
                    ),
                )
            )
        except StopIteration:
            raise RuntimeError(
                f"Could not find any problem (train/val/test) that matches the name {name}"
            )

    @staticmethod
    def problem_matching_name(name: str, problems: Iterable[XProblem] | None):
        return (
            next(
                (p for p in problems if name in p.name),
                None,
            )
            if problems is not None
            else None
        )

    @cache
    def _problem_checkpoints(
        self, probs_store_callback_name="actor_probs"
    ) -> dict[XProblem, list[tuple[int, Path]]]:
        probs_dir = self.out_data.out_dir / probs_store_callback_name
        if not probs_dir.is_dir():
            raise OSError(
                f"Could not find actor_probs for experiment at {self.out_data.out_dir}. "
                f"Path does not lead to a directory."
            )
        probs_paths = list(probs_dir.iterdir())
        by_problem: dict[XProblem, list[tuple[int, Path]]] = defaultdict(list)
        for path in probs_paths:
            stem = path.stem.removeprefix(f"{probs_store_callback_name}_")
            epoch: str = stem.split("_")[-1]
            dataloader_name = stem.removesuffix(f"_{epoch}")
            problem: XProblem = self.find_problem_by_name(dataloader_name)
            by_problem[problem].append((int(epoch), path))
        by_problem = {p: sorted(paths) for (p, paths) in by_problem.items()}
        return by_problem

    @cache
    def stored_probs(
        self, problem: XProblem, probs_store_callback_name="actor_probs"
    ) -> list[tuple[int, list[torch.Tensor]]]:
        epoch_list = self._problem_checkpoints(probs_store_callback_name).get(
            problem, []
        )
        if len(epoch_list) == 0:  # e.g., no validation epoch finished
            return []

        assert all(
            isinstance(epoch, int) and isinstance(path, Path)
            for epoch, path in epoch_list
        ), "Expected epoch_list to contain tuples of (epoch, Path)"

        loaded_probs_list: list[tuple[int, list[torch.Tensor]]] = [
            (epoch, torch.load(path, map_location=self.device))
            for epoch, path in epoch_list
        ]
        return loaded_probs_list
