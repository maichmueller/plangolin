import logging
import os
import shelve
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, Union

import torch
from torch_geometric.data import Batch, Data, HeteroData, InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.separate import separate
from torch_geometric.io import fs

import xmimir as xmi
from rgnet.encoding.base_encoder import EncoderFactory
from rgnet.rl.envs import ExpandedStateSpaceEnv, PlanningEnvironment
from rgnet.rl.envs.expanded_state_space_env import (
    ExpandedStateSpaceEnvLoader,
    IteratingReset,
)
from rgnet.rl.reward import RewardFunction
from rgnet.utils.misc import persistent_hash
from xmimir import XProblem, XStateSpace, parse


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
        self.problem: XProblem | None = None
        self.encoder_factory = encoder_factory
        self.encoder = None
        self.reward_function = reward_function or getattr(env, "reward_function")
        self._maybe_env: PlanningEnvironment | ExpandedStateSpaceEnvLoader | None = env
        self._env: PlanningEnvironment | None = None
        self._space: XStateSpace | None = None
        self.device = device
        self.space_options = space_options
        self.show_progress = show_progress
        self._data_cache = dict()
        self.logging_kwargs = logging_kwargs  # will be removed after process(), otherwise pickling not possible
        metadata_hash = persistent_hash(self.metadata.values())
        root_dir = Path(root_dir) / metadata_hash
        # initialize klepto store for modular persistence
        os.makedirs(root_dir / "database", exist_ok=True)
        self.metabase = shelve.open(str(root_dir / "database"), flag="c")
        # verify that the metadata matches the current configuration; otherwise we cannot trust previously processed
        # data will align with our expectations.
        self.desc: Optional[str] = None

        if metadata := self.try_get_data("meta"):
            self.desc = metadata.get("desc", None)
            if mismatch_desc := self._metadata_misaligned(metadata):
                logging.info(
                    f"Metadata mismatch ({mismatch_desc}) for problem {self.problem_path}, forcing reload."
                )
                force_reload = True

        super().__init__(
            root=str(root_dir.absolute()),
            transform=transform,
            pre_transform=None,
            pre_filter=None,
            log=log,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    def try_get_data(self, key: str) -> dict[str, Any] | None:
        r"""Attempts to retrieve metadata from the metabase."""
        try:
            if key in self._data_cache:
                # if we have already loaded this key, return the cached value
                return self._data_cache[key]
            data = self.metabase[key]
            self._data_cache[key] = data
            return data
        except KeyError:
            keytree = self._metabase_keytree()
            key_parts = key.split(".")
            for part in key_parts:
                if part not in keytree:
                    return None
                keytree = keytree[part]
            return self._retrieve_data_from_keytree(key_parts, keytree)

    def _metabase_keytree(self):
        """
        Build a nested dict of the key hierarchy without loading any values.
        Leaves are marked as None.
        """
        tree = {}
        for full_key in self.metabase.keys():
            parts = full_key.split(".")
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # mark last part as a leaf with None
            current[parts[-1]] = None
        return tree

    def _retrieve_data_from_keytree(
        self, key_parts: List[str], keytree: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Given:
          key_parts — the path down to this subtree (as list of fragments),
          keytree   — the subtree dict where leaves are None,
        returns a nested dict of the same shape, but with each leaf replaced
        by self.metabase['.'.join(full path)].
        """
        result: dict[str, Any] = {}
        for part, subtree in keytree.items():
            full_path = key_parts + [part]
            full_key = ".".join(full_path)
            if subtree is None:
                # leaf: load the actual value
                result[part] = self.metabase.get(full_key)
            else:
                # branch: recurse deeper
                result[part] = self._retrieve_data_from_keytree(full_path, subtree)
        return result

    @property
    def env(self) -> PlanningEnvironment | ExpandedStateSpaceEnv | None:
        if self._env is not None:
            return self._env

        if self._maybe_env is not None:
            env = self._maybe_env
            self._maybe_env = None  # clear the maybe_env to avoid reloading
            if isinstance(env, ExpandedStateSpaceEnvLoader):
                try:
                    env = env()
                except KeyError:
                    # this is not a problem for which we can load the environment,
                    # perhaps the space is too large or the environment is not available.
                    # We continue without, perhaps a deriving drive class can handle this.
                    env = None
            elif not isinstance(env, PlanningEnvironment):
                raise ValueError(
                    f"`env` is not an instance of {PlanningEnvironment.__class__}. Given: {type(env)}"
                )
            if env is None:
                logging.warning(
                    "ExpandedStateSpaceEnvLoader returned None, "
                    "perhaps the space is too large."
                )
            self._env = env
            return self._env
        space = self._make_space()
        self._space = space
        env = ExpandedStateSpaceEnv(
            space,
            batch_size=torch.Size((len(space),)),
            reset_strategy=IteratingReset(),
            reward_function=self.reward_function,
            reset=True,
        )
        self._env = env
        return self._env

    def __str__(self):
        return self.desc

    def _load(self):
        r"""Loads the dataset from the processed file."""
        if not self.processed_paths:
            raise RuntimeError(
                "Dataset not processed yet. Call `process()` before loading."
            )
        # avoid pickling un-pickleable pymimir objects by simply removing it
        del self._env
        del self._maybe_env
        del self._space
        del self.problem
        self.load(self.processed_paths[0])

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

    def _metadata_misaligned(self, meta: dict) -> str:
        if self.__class__ != meta["class_"]:
            return f"Class: given={self.__class__} != loaded={meta['class_']}"
        if (
            self.encoder_factory is not None
            and self.encoder_factory != meta["encoder_factory"]
        ):
            return f"encoder_factory: given={self.encoder_factory} != loaded={meta['encoder_factory']}"
        if self.reward_function != meta["reward_function"]:
            return f"reward_function: given={self.reward_function} != loaded={meta['reward_function']}"
        if (our_file := self.domain_path.read_text()) != meta["domain_content"]:
            return f"domain: given={our_file} != loaded={meta['domain_content']}"
        if (our_file := self.problem_path.read_text()) != meta["problem_content"]:
            return f"problem: given={our_file} != loaded={meta['problem_content']}"
        if self.space_options != meta["space_options"]:
            return f"space_options: given={self.space_options} != loaded={meta['space_options']}"
        return ""

    @property
    def metadata(self) -> dict:
        return dict(
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

        space = self._make_space()
        data_list = self._build()
        self._set_desc(space)
        self._save_metadata(desc=self.desc)

        self._get_logger().info(
            f"Saving {self.__class__.__name__} "
            f"(problem: {self.problem.name} / {Path(self.problem.filepath).stem}, space: {space})"
        )
        del self.logging_kwargs
        self.save(data_list, self.processed_paths[0])

    def env_aux_data(self) -> dict:
        if pyg_env := self.try_get_data("aux.pyg_env"):
            # if we have already processed the environment, we can simply return the cached data
            return pyg_env
        env = self.env
        space = env.active_instances[0]
        logger = self._get_logger()
        logger.info(
            f"Auxiliary Data ({BaseDrive.__name__}: "
            f"problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        pyg_env = env.to_pyg_data(0)
        logger.info(f"Auxiliary Data ({BaseDrive.__name__}): Finished.")
        self._save_aux_data(pyg_env=pyg_env)
        return self.try_get_data("aux.pyg_env")

    def _save_aux_data(self, **aux_data):
        r"""Saves auxiliary data to the metabase."""
        for key, value in aux_data.items():
            self.metabase[f"aux.{key}"] = value

    def _save_metadata(self, **extras):
        metad = self.metadata | extras
        assert not callable(metad), (
            "Metadata must not be a callable, "
            "it should be a dataclass/dict instance."
        )
        try:
            for key, value in metad.items():
                self.metabase[f"meta.{key}"] = value
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.metabase.clear()
            raise RuntimeError(
                f"Failed to save metadata to metabase at {Path(self.metabase.archive.name)}. "
                "Please check the file permissions or disk space."
            ) from e

    def _set_desc(self, space: xmi.XStateSpace | None):
        if space is None:
            if hasattr(self, "problem"):
                problem = self.problem
            else:
                _, problem = parse(self.domain_path, self.problem_path)
        else:
            problem = space.problem
        self.desc = f"{self.__class__.__name__}({problem.name}, {problem.filepath}, state_space={str(space)})"

    def _space_from_env(self):
        r"""Returns the state space from the environment."""
        if self.env is None:
            raise RuntimeError(
                "Environment is not set. Please set `env` before calling this method."
            )
        return self.env.active_instances[0]

    def _make_space(self):
        if self._space is not None:
            return self._space
        space = xmi.XStateSpace(
            str(self.domain_path.absolute()),
            str(self.problem_path.absolute()),
            **(self.space_options or dict()),
        )
        return space

    @abstractmethod
    def _build(self) -> List[HeteroData]: ...

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
        if isinstance(data, (Data, HeteroData)) and key in data:
            if self._indices is None and data.__inc__(key, data[key]) == 0:
                return data[key]
            else:
                data_list = [self.get(i) for i in self.indices()]
                return Batch.from_data_list(data_list)[key]
        else:
            return object.__getattribute__(self, key)
