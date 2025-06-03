import logging
import pickle
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from os.path import splitext
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, Union

import torch
from torch_geometric.data import Batch, Data, HeteroData, InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.separate import separate
from torch_geometric.io import fs

import xmimir as xmi
from rgnet.encoding.base_encoder import EncoderFactory
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.envs.expanded_state_space_env import (
    ExpandedStateSpaceEnvLoader,
    IteratingReset,
)
from rgnet.rl.reward import RewardFunction
from rgnet.utils.misc import persistent_hash


@dataclass(frozen=True)
class BaseDriveMetadata:
    class_: type
    encoder_factory: EncoderFactory
    reward_function: RewardFunction
    domain_content: str
    problem_content: str
    space_options: Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class BaseEnvAuxData:
    pyg_env: Data


class BaseDrive(InMemoryDataset):
    def __init__(
        self,
        root_dir: Path | str,
        domain_path: Path | None = None,
        problem_path: Path | None = None,
        env: ExpandedStateSpaceEnv | ExpandedStateSpaceEnvLoader | None = None,
        reward_function: RewardFunction | None = None,
        encoder_factory: Optional[EncoderFactory] = None,
        *,
        device: str | torch.device | None = None,
        transform: Callable[[HeteroData | Data], HeteroData | Data] = None,
        save_aux_data: bool = True,
        log: bool = False,
        force_reload: bool = False,
        show_progress: bool = True,
        logging_kwargs: Optional[Mapping[str, Any]] = None,
        space_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.domain_path: Path = domain_path
        self.problem_path: Path = problem_path
        if self.domain_path is None and env is not None:
            assert not isinstance(env, ExpandedStateSpaceEnvLoader), (
                "`env` must be an instance of ExpandedStateSpaceEnv "
                "if not providing `problem_path` and `domain_path`."
            )
            self.domain_path = Path(env.active_instances[0].problem.domain.filepath)
        if self.problem_path is None and env is not None:
            assert not isinstance(env, ExpandedStateSpaceEnvLoader), (
                "`env` must be an instance of ExpandedStateSpaceEnv "
                "if not providing `problem_path` and `domain_path`."
            )
            self.problem_path = Path(env.active_instances[0].problem.filepath)
        assert (
            self.domain_path is not None and self.problem_path is not None
        ), "Domain or problem paths are None."
        assert self.domain_path.exists() and self.domain_path.is_file()
        assert self.problem_path.exists() and self.problem_path.is_file()
        self.encoder_factory = encoder_factory
        self.encoder = None
        self.reward_function = reward_function or getattr(env, "reward_function")
        self.desc: Optional[str] = None
        self.env = env
        self.device = device
        self.space_options = space_options
        self.save_aux_data = save_aux_data
        self.show_progress = show_progress
        self.logging_kwargs = logging_kwargs  # will be removed after process(), otherwise pickling not possible
        metadata_hash = persistent_hash(self.metadata.__dict__.values())
        root_dir = Path(root_dir) / metadata_hash
        self.metadata_path: Path = Path(root_dir) / (
            str(splitext(self.processed_file_names[0])[0]) + ".meta.pt"
        )
        if self.metadata_path.exists():
            # verify that the metadata matches the current configuration; otherwise we cannot trust previously processed
            # data will align with our expectations.
            loaded_metadata, self.desc, *_ = self._load_metadata()
            if mismatch_desc := self._metadata_misaligned(loaded_metadata):
                logging.info(
                    f"Metadata mismatch ({mismatch_desc}) for problem {self.problem_path}, forcing reload."
                )
                force_reload = True
        if self.save_aux_data:
            self.aux_data_path = Path(root_dir) / (
                str(splitext(self.processed_file_names[0])[0]) + ".graph.pt"
            )
        else:
            self.aux_data_path = None

        super().__init__(
            root=str(root_dir.absolute()),
            transform=transform,
            pre_transform=None,
            pre_filter=None,
            log=log,
            force_reload=force_reload,
        )
        del (
            self.env
        )  # avoid pickling the un-pickleable env object by simply removing it
        self.load(self.processed_paths[0])

    def __str__(self):
        return self.desc

    def load(self, path: str, data_cls: Type[BaseData] = Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        out = fs.torch_load(path, map_location=self.device or "cpu")
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    @property
    def env_aux_data(self):
        if self.aux_data_path is not None and self.aux_data_path.exists():
            with open(self.aux_data_path, "rb") as file:
                aux_data = pickle.load(file)
            return aux_data
        warnings.warn(
            f"Drive object has no auxiliary data path set ({self.aux_data_path = }) or "
            f"the data does not exist ({self.aux_data_path.exists() = }. "
        )
        return None

    def _metadata_misaligned(self, meta: BaseDriveMetadata) -> str:
        if self.__class__ != meta.class_:
            return f"Class: given={self.__class__} != loaded={meta.class_}"
        if (
            self.encoder_factory is not None
            and self.encoder_factory != meta.encoder_factory
        ):
            return f"encoder_factory: given={self.encoder_factory} != loaded={meta.encoder_factory}"
        if self.reward_function != meta.reward_function:
            return f"reward_function: given={self.reward_function} != loaded={meta.reward_function}"
        if (our_file := self.domain_path.read_text()) != meta.domain_content:
            return f"domain: given={our_file} != loaded={meta.domain_content}"
        if (our_file := self.problem_path.read_text()) != meta.problem_content:
            return f"problem: given={our_file} != loaded={meta.problem_content}"
        if self.space_options != meta.space_options:
            return f"space_options: given={self.space_options} != loaded={meta.space_options}"
        return ""

    @property
    def metadata(self) -> BaseDriveMetadata:
        return BaseDriveMetadata(
            class_=self.__class__,
            encoder_factory=self.encoder_factory,
            reward_function=self.reward_function,
            domain_content=self.domain_path.read_text(),
            problem_content=self.problem_path.read_text(),
            space_options=self.space_options,
        )

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [f"{self.problem_path.stem}.pt"]

    def process(self) -> None:
        """
        Process the domain and problem files to build the dataset.
        Only called if self.force_reload is True or the processed dataset file does not exist.
        """
        if self.encoder_factory is None:
            raise ValueError(
                "Encoder factory must be provided when data has not been processed prior."
            )
        if self.env is not None:
            env = self.env
            if isinstance(env, ExpandedStateSpaceEnvLoader):
                env = env()
            if not isinstance(env, ExpandedStateSpaceEnv):
                raise ValueError(
                    f"`env` is not an instance of ExpandedStateSpaceEnv. Given: {type(env)}"
                )
            space = env.active_instances[0]
        else:
            space = self._make_space()
            env = ExpandedStateSpaceEnv(
                space,
                batch_size=torch.Size((len(space),)),
                reset_strategy=IteratingReset(),
                reward_function=self.reward_function,
                reset=True,
            )
        self._set_desc(space)

        if self.save_aux_data:
            aux = self._make_env_aux_data(env)
            with open(self.aux_data_path, "wb") as file:
                pickle.dump(aux, file)

        data_list = self._build(env)

        self._save_metadata(self.desc)

        self._get_logger().info(
            f"Saving {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        del self.logging_kwargs
        self.save(data_list, self.processed_paths[0])

    def _make_env_aux_data(self, env: ExpandedStateSpaceEnv) -> BaseEnvAuxData:
        space = env.active_instances[0]
        logger = self._get_logger()
        logger.info(
            f"Auxiliary Data ({BaseDrive.__name__}: "
            f"problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        data = BaseEnvAuxData(env.to_pyg_data(0))
        logger.info(f"Auxiliary Data ({BaseDrive.__name__}): Finished.")
        return data

    def _save_metadata(self, *extras):
        with open(self.metadata_path, "wb") as file:
            pickle.dump((self.metadata, *extras), file)

    def _load_metadata(
        self,
    ) -> tuple[BaseDriveMetadata] | tuple[BaseDriveMetadata, Any, ...]:
        with open(self.metadata_path, "rb") as file:
            loaded_metadata, *extras = pickle.load(file)

        return (loaded_metadata, *extras)

    def _set_desc(self, space: xmi.XStateSpace):
        self.desc = f"{self.__class__.__name__}({space.problem.name}, {space.problem.filepath}, state_space={str(space)})"

    def _make_space(self):
        space = xmi.XStateSpace(
            str(self.domain_path.absolute()),
            str(self.problem_path.absolute()),
            **(self.space_options or dict()),
        )
        return space

    @abstractmethod
    def _build(self, env: ExpandedStateSpaceEnv) -> List[HeteroData]: ...

    def _get_logger(self):
        if self.logging_kwargs is not None:
            logger = logging.getLogger(
                f"{self.__class__.__name__}-{self.logging_kwargs['task_id']}"
            )
            logger.setLevel(self.logging_kwargs["log_level"])
        else:
            logger = logging.root
        return logger

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
