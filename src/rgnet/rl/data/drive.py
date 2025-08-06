import logging
import os
import shelve
import shutil
from abc import abstractmethod
from contextlib import contextmanager
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
from rgnet.logging_setup import get_logger
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


logger = get_logger(__name__)


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
        cache_items: bool = False,
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
        # whether to cache access to the actual data of the dataset (as the base class does)
        self.cache_items = cache_items
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
        self.metabase_path = root_dir / "database.db"
        os.makedirs(root_dir, exist_ok=True)
        # copy over the used domain and problem files to the root directory
        shutil.copy(self.domain_path, root_dir / "domain.pddl")
        shutil.copy(self.problem_path, root_dir / "problem.pddl")
        self.metabase: shelve.Shelf | None = None
        # verify that the metadata matches the current configuration; otherwise we cannot trust previously processed
        # data will align with our expectations.
        self.desc: Optional[str] = None
        if (metadata := self.try_get_data("meta")) is not None:
            # desc is not explicit part of metadata, but also stored there anyway
            self.desc = metadata["desc"]
            if mismatch_desc := self._metadata_misaligned(metadata):
                logger.info(
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

    def try_open_metabase(self, flag="c") -> bool:
        """
        Close any existing shelf, then open a fresh one with the given flag.
        """
        # Close previous shelf if open
        if self.metabase is not None:
            try:
                self.metabase.close()
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
        try:
            self.metabase = shelve.open(str(self.metabase_path), flag=flag)
            return True
        except KeyboardInterrupt:
            raise
        except Exception:
            try:
                self.metabase = shelve.open(str(self.metabase_path), flag="c")
                return True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                get_logger(__name__).error(
                    f"Failed to open metabase at {self.metabase_path}: {e}"
                )
                return False

    def try_get_data(self, key: str) -> Any | dict[str, Any] | None:
        r"""Attempts to retrieve metadata from the metabase."""
        if key in self._data_cache:
            # if we have already loaded this key, return the cached value
            return self._data_cache[key]
        with self.metabase_open("r"):
            try:
                data = self.metabase[key]
            except KeyError:
                keytree = self._metabase_keytree()
                key_parts = key.split(".")
                for part in key_parts:
                    if part not in keytree:
                        return None
                    keytree = keytree[part]
                data = self._retrieve_data_from_keytree(key_parts, keytree)
        self._data_cache[key] = data
        return data

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
                logger.warning(
                    "ExpandedStateSpaceEnvLoader returned None, "
                    "perhaps the space is too large."
                )
            self._env = env
            return self._env
        space = self.get_space()
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

    @contextmanager
    def metabase_open(self, flag="c"):
        """
        Context‐manager that opens a fresh shelf with the given flag,
        yields the shelf object, and ensures it’s closed afterward.
        """
        # Close any existing handle
        if self.metabase is not None:
            try:
                self.metabase.close()
            except Exception:
                pass
        # Open a new shelf instance
        success = self.try_open_metabase(flag=flag)
        try:
            yield success
        finally:
            self.metabase.close()
            self.metabase = None

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

        data_list = self._build()
        self._set_desc()
        self._save_metadata(desc=self.desc)
        # save auxiliary data to the database
        self.env_aux_data()

        self._get_logger().info(
            f"Saving {self.__class__.__name__} "
            f"(problem: {self.problem.name} / {Path(self.problem.filepath).stem}, space: {self.get_space()})"
        )
        del self.logging_kwargs
        self.save(data_list, self.processed_paths[0])

    def env_aux_data(self) -> dict:
        """
        Retrieves auxiliary data from the environment, such as the PyG representation of the environment.

        If the auxiliary data has already been processed and saved, it will return the cached data.
        Otherwise, it will process the environment to generate the auxiliary data and save it to the metabase.
        """
        if (pyg_env := self.try_get_data("aux.pyg_env")) is not None:
            # if we have already processed the environment, we can simply return the cached data
            ...
        else:
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
        return {"pyg_env": pyg_env}

    def _save_aux_data(self, **aux_data):
        r"""Saves auxiliary data to the metabase."""
        with self.metabase_open(flag="w"):
            for key, value in aux_data.items():
                self.metabase[f"aux.{key}"] = value

    def _save_metadata(self, **extras):
        metad = self.metadata | extras
        assert not callable(metad), (
            "Metadata must not be a callable, "
            "it should be a dataclass/dict instance."
        )
        with self.metabase_open(flag="w"):
            for key, value in metad.items():
                self.metabase[f"meta.{key}"] = value

    def _set_desc(self):
        if hasattr(self, "problem"):
            problem = self.problem
        else:
            _, problem = parse(self.domain_path, self.problem_path)
        self.desc = f"{self.__class__.__name__}({problem.name}, {problem.filepath}, state_space={str(self.get_space())})"
        assert self.metabase is None
        with self.metabase_open(flag="w"):
            assert self.metabase is not None
            self.metabase["desc"] = self.desc

    def _space_from_env(self):
        r"""Returns the state space from the environment."""
        if self.env is None:
            raise RuntimeError(
                "Environment is not set. Please set `env` before calling this method."
            )
        return self.env.active_instances[0]

    def get_space(self):
        if self._space is not None:
            return self._space
        space = xmi.XStateSpace(
            str(self.domain_path.absolute()),
            str(self.problem_path.absolute()),
            **(self.space_options or dict()),
        )
        self._space = space
        return space

    @abstractmethod
    def _build(self) -> List[HeteroData]: ...

    def _get_logger(self):
        if self.logging_kwargs is not None:
            l = logging.getLogger(f"{__name__}-thread:{self.logging_kwargs['task_id']}")
            l.setLevel(self.logging_kwargs["log_level"])
            return l
        else:
            return logger

    def get(self, idx: int) -> BaseData:
        """
        Get the data object at the given index.

        NOTE:
            Overrides the base-method to avoid caching previously fetched datapoints and increasing memory usage
            without gain. If `self.cache_items` is True, it will use the base class method as normal.
        """
        if self.cache_items:
            return super().get(idx)
        return separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

    def __getstate__(self):
        """
        Custom __getstate__ to avoid pickling the metabase and other non-pickleable attributes.
        """
        state = self.__dict__.copy()
        # remove non-pickleable attributes
        state["metabase"] = None
        state["_data_cache"].clear()
        state.pop("_maybe_env", None)
        state.pop("_env", None)
        state.pop("_space", None)
        state.pop("problem", None)
        state.pop("logging_kwargs", None)
        return state

    def __setstate__(self, state):
        """
        Custom __setstate__ to restore the metabase and other attributes.
        """
        self.__dict__.update(state)
        # reinitialize the metabase
        logger.warning("state used:", state.items())
        # reinitialize the problem if it was not set
        _, self.problem = parse(self.domain_path, self.problem_path)

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
