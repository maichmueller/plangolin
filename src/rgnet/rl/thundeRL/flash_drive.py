import logging
import pickle
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
from torch_geometric.data import Batch, Data, HeteroData, InMemoryDataset
from torch_geometric.data.separate import separate
from tqdm import tqdm

import xmimir as xmi
from rgnet.encoding import GraphEncoderBase
from rgnet.encoding.base_encoder import EncoderFactory
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.envs.expanded_state_space_env import IteratingReset
from rgnet.rl.reward import RewardFunction, UnitReward


class FlashDrive(InMemoryDataset):
    def __init__(
        self,
        domain_path: Path,
        problem_path: Path,
        reward_function: RewardFunction = UnitReward(gamma=0.9),
        encoder_factory: Optional[EncoderFactory] = None,
        max_expanded: Optional[int] = None,
        root_dir: Optional[str] = None,
        log: bool = False,
        force_reload: bool = False,
        show_progress: bool = True,
        logging_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        assert domain_path.exists() and domain_path.is_file()
        assert problem_path.exists() and problem_path.is_file()
        self.desc: Optional[str] = None
        self.domain_file: Path = domain_path
        self.problem_path: Path = problem_path
        self.encoder_factory = encoder_factory
        self.reward_function = reward_function
        self.max_expanded = max_expanded
        self.show_progress = show_progress
        self.logging_kwargs = logging_kwargs  # will be removed after process(), otherwise pickling not possible
        self.metadata_path = Path(root_dir) / (
            str(self.processed_file_names[0]) + ".meta"
        )
        if self.metadata_path.exists():
            # verify that the metadata matches the current configuration, otherwise we cannot trust previously processed
            # data will align with our expectations.
            with open(self.metadata_path, "rb") as file:
                loaded_metadata = pickle.load(file)
            if mismatch_desc := self._metadata_misaligned(loaded_metadata):
                logging.info(
                    f"Metadata mismatch ({mismatch_desc}) for problem {self.problem_path}, forcing reload."
                )
                force_reload = True
            else:
                self.desc = loaded_metadata[0]
        super().__init__(
            root=root_dir,
            transform=self.target_idx_to_data_transform,
            pre_transform=None,
            pre_filter=None,
            log=log,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    def __str__(self):
        return self.desc

    def _metadata_misaligned(self, meta: Tuple) -> str:
        (
            _,
            encoder_factory,
            reward_function,
            domain_content,
            problem_content,
            max_expanded,
        ) = meta
        if self.encoder_factory is not None and self.encoder_factory != encoder_factory:
            return f"encoder_factory: given={self.encoder_factory} != loaded={encoder_factory}"
        if self.reward_function != reward_function:
            return f"reward_function: given={self.reward_function} != loaded={reward_function}"
        if (our_file := self.domain_file.read_text()) != domain_content:
            return f"domain: given={our_file} != loaded={domain_content}"
        if (our_file := self.problem_path.read_text()) != problem_content:
            return f"problem: given={our_file} != loaded={problem_content}"
        if self.max_expanded != max_expanded:
            return f"max_expanded: given={self.max_expanded} != loaded={max_expanded}"
        return ""

    @property
    def metadata(self):
        domain_content = self.domain_file.read_text()
        problem_content = self.problem_path.read_text()
        return (
            self.desc,
            self.encoder_factory,
            self.reward_function,
            domain_content,
            problem_content,
            self.max_expanded,
        )

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
        if self.encoder_factory is None:
            raise ValueError(
                "Encoder factory must be provided when data has not been processed prior."
            )
        space = xmi.XStateSpace(
            str(self.domain_file.absolute()),
            str(self.problem_path.absolute()),
            max_num_states=self.max_expanded or 1_000_000,
        )
        env = ExpandedStateSpaceEnv(
            space,
            batch_size=torch.Size((len(space),)),
            reset_strategy=IteratingReset(),
            reward_function=self.reward_function,
        )
        data_list = self._build(
            env,
            self.encoder_factory(space.problem.domain),
        )
        with open(self.metadata_path, "wb") as file:
            pickle.dump(self.metadata, file)
        self._get_logger().info(
            f"Saving {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        del self.logging_kwargs
        self.save(data_list, self.processed_paths[0])

    def _build(
        self,
        env: ExpandedStateSpaceEnv,
        encoder: GraphEncoderBase,
    ) -> List[HeteroData]:
        out = env.reset(autoreset=False)
        space: xmi.XStateSpace = out[env.keys.instance][0]
        self.desc = f"FlashDrive({space.problem.name}, {space.problem.filepath}, state_space={str(space)})"
        nr_states: int = len(space)
        logger = self._get_logger()
        logger.info(
            f"Building {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        # Each data object represents one state
        batched_data: List[Union[HeteroData, Data]] = [None] * nr_states
        space_iter = zip(out["state"], out["transitions"], out["instance"])
        if self.show_progress:
            space_iter = tqdm(space_iter, total=nr_states, desc="Encoding states")
        for state, transitions, instance in space_iter:
            data = encoder.to_pyg_data(encoder.encode(state))
            reward, done = env.get_reward_and_done(
                transitions, instances=[instance] * len(transitions)
            )
            data.reward = reward
            # Save the index of the state
            # NOTE: No element should contain the attribute `index`, as it is used by PyG internally.
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html

            # this index needs to be guaranteed to be the same each time a StateSpace is created from the same problem.
            # We need to verify if this is the case for the current implementation of pymimir.
            data.idx = state.index
            data.done = done
            # Same index concerns for transition.target.index
            data.targets = list(t.target.index for t in transitions)
            distance_to_goal = space.goal_distance(state)
            if distance_to_goal == float("inf"):
                # deadend states receive label -1
                distance_to_goal = -1
            data.distance_to_goal = torch.tensor(distance_to_goal, dtype=torch.long)
            batched_data[state.index] = data
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        return batched_data

    def _get_logger(self):
        if self.logging_kwargs is not None:
            logger = logging.getLogger(f"FD-{self.logging_kwargs['task_id']}")
            logger.setLevel(self.logging_kwargs["log_level"])
        else:
            logger = logging.root
        return logger

    def target_idx_to_data_transform(
        self, data: Union[HeteroData, Data]
    ) -> Union[HeteroData, Data]:
        """
        Convert transition target state indices to actual hetero-data objects.
        :param data the hetero-data object to transform.
        :returns The transformed hetero-data object.
        """
        data.targets = tuple(
            self.get(target) if isinstance(target, int) else target
            for target in data.targets
        )
        return data

    def get(self, idx: int) -> Union[HeteroData, Data]:
        """
        Get the data object at the given index.

        NOTE:
            Override the base-method to avoid caching previously fetched datapoints
             and increasing memory usage without gain.
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
