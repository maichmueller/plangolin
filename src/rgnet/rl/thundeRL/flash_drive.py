import logging
import warnings
from logging import handlers
from multiprocessing import Lock
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import pymimir as mi
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from torch_geometric.data.separate import separate
from tqdm import tqdm

from rgnet.encoding import HeteroGraphEncoder
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.envs.expanded_state_space_env import IteratingReset

# TODO: Remove this once the issue is resolved in pytorch-geometric
warnings.filterwarnings("ignore")


class FlashDrive(InMemoryDataset):
    def __init__(
        self,
        domain_path: Path,
        problem_path: Path,
        custom_dead_end_reward: float,
        max_expanded: Optional[int] = None,
        root_dir: Optional[str] = None,
        log: bool = False,
        force_reload: bool = False,
        show_progress: bool = True,
        logging_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        assert domain_path.exists() and domain_path.is_file()
        assert problem_path.exists() and problem_path.is_file()
        self.domain_file: Path = domain_path
        self.problem_path: Path = problem_path
        self.custom_dead_end_reward = custom_dead_end_reward
        self.max_expanded = max_expanded
        self.show_progress = show_progress
        self.logging_kwargs = logging_kwargs  # will be removed after process()
        super().__init__(
            root=root_dir,
            transform=self.target_idx_to_data_transform,
            pre_transform=None,
            pre_filter=None,
            log=log,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [self.problem_path.stem + ".pt"]

    def process(self) -> None:
        """
        Process the domain and problem files to build the dataset.

        Only called if self.force_reload is True or the processed dataset file does not exist.
        """
        domain = mi.DomainParser(str(self.domain_file.absolute())).parse()
        problem = mi.ProblemParser(str(self.problem_path.absolute())).parse(domain)
        space = mi.StateSpace.new(
            problem,
            mi.GroundedSuccessorGenerator(problem),
            self.max_expanded or 1_000_000,
        )
        env = ExpandedStateSpaceEnv(
            space,
            batch_size=torch.Size((1,)),
            reset_strategy=IteratingReset(),
            custom_dead_end_reward=self.custom_dead_end_reward,
        )
        data_list = self._build(env, HeteroGraphEncoder(domain))
        self.save(data_list, self.processed_paths[0])

    def _build(
        self,
        env: ExpandedStateSpaceEnv,
        encoder: HeteroGraphEncoder,
    ) -> List[HeteroData]:
        out = env.reset()
        space = out[env.keys.instance][0]
        nr_states = space.num_states()
        self._log_build_start(space)
        # Each data object represents one state
        batched_data: List[HeteroData] = [None] * nr_states
        state_to_idx = {state: i for i, state in enumerate(space.get_states())}
        state_iter = state_to_idx.items()
        if self.show_progress:
            state_iter = tqdm(state_iter, total=nr_states, desc="Encoding states")
        for state, i in state_iter:
            data = encoder.to_pyg_data(encoder.encode(state))
            transitions = space.get_forward_transitions(state)
            reward, done = env.get_reward_and_done(
                actions=transitions,
                current_states=[t.source for t in transitions],
                instances=[space] * len(transitions),
            )
            data.reward = reward
            # Save the index of the state
            # NOTE: No element should contain index
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html
            data.idx = i
            data.done = done
            data.targets = tuple(
                state_to_idx[transition.target] for transition in transitions
            )
            batched_data[i] = data
        return batched_data

    def _log_build_start(self, space):
        if self.logging_kwargs is not None:
            logger = logging.getLogger(f"thread-{self.logging_kwargs['thread_id']}")
        else:
            logger = logging.getLogger("root")
        logger.setLevel(self.logging_kwargs["log_level"])
        logger.info(
            f"Building {self.__class__.__name__} "
            f"(problem: {space.problem.name}, #states: {space.num_states()})"
        )
        del self.logging_kwargs

    def target_idx_to_data_transform(self, data: HeteroData) -> HeteroData:
        """
        Convert transition target state indices to actual hetero-data objects.

        Parameters
        ----------
        data: HeteroData,
            The hetero-data object to transform.
        Returns
        -------
        HeteroData
            The transformed hetero-data object.
        """
        data.targets = tuple(
            self.get(target) if isinstance(target, int) else target
            for target in data.targets
        )
        return data

    def get(self, idx: int) -> HeteroData:
        """
        Get the data object at the given index.

        Override the base-method to avoid caching previously fetched datapoints and increasing memory usage without gain.
        """
        return separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

    def __getattr__(self, key: str) -> Any:
        """
        InMemoryDataset forgot the poor HeteroData objects, logic is equivalent.
        """
        data = self.__dict__.get("_data")
        if isinstance(data, HeteroData) and key in data:
            if self._indices is None and data.__inc__(key, data[key]) == 0:
                return data[key]
            else:
                data_list = [self.get(i) for i in self.indices()]
                return Batch.from_data_list(data_list)[key]
        else:
            return super().__getattr__(key)
