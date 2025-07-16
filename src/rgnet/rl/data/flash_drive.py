import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

from torch_geometric.data import Data, HeteroData

import xmimir as xmi
from rgnet.logging_setup import get_logger, tqdm
from rgnet.rl.envs import ExpandedStateSpaceEnv
from xmimir import ActionHistoryDataPack, XState, XTransition, gather_objects
from xmimir.iw import IWSearch, IWStateSpace, RandomizedExpansion

from .drive import BaseDrive, BaseDriveMetadata


class attr_getters:
    """
    Namespaced attribute getters
    """

    @staticmethod
    def idx(
        drive: "FlashDrive",
        env: ExpandedStateSpaceEnv,
        state: XState,
        transitions: Sequence[XTransition],
    ) -> int:
        return state.index

    @staticmethod
    def object_count(
        drive: "FlashDrive",
        env: ExpandedStateSpaceEnv,
        state: XState,
        transitions: Sequence[XTransition],
    ) -> int:
        return len(gather_objects(state))

    @staticmethod
    def targets(
        drive: "FlashDrive",
        env: ExpandedStateSpaceEnv,
        state: XState,
        transitions: Sequence[XTransition],
    ) -> list[int]:
        return [t.target.index for t in transitions]

    @staticmethod
    def distance_to_goal(
        drive: "FlashDrive",
        env: ExpandedStateSpaceEnv,
        state: XState,
        transitions: Sequence[XTransition],
    ) -> float:
        space = env.active_instances[0]
        dist = space.goal_distance(state)
        if dist == float("inf"):
            dist = -1
        return dist

    @staticmethod
    def action_history_datapack(
        drive: "FlashDrive",
        env: ExpandedStateSpaceEnv,
        state: XState,
        transitions: Sequence[XTransition],
    ) -> ActionHistoryDataPack:
        # shortest_dists = env.active_instances[
        #     0
        # ].compute_pairwise_shortest_backward_state_distances()
        return ActionHistoryDataPack(
            (
                t.action
                for t in env.active_instances[0].a_star_search(
                    target=state,
                    # dists=shortest_dists[state.index],
                    forward=False,
                )
            ),
        )

    @staticmethod
    def problem_path(
        drive: "FlashDrive",
        env: ExpandedStateSpaceEnv,
        state: XState,
        transitions: Sequence[XTransition],
    ) -> str:
        return str(Path(drive.root) / "problem.pddl")

    @staticmethod
    def domain_path(
        drive: "FlashDrive",
        env: ExpandedStateSpaceEnv,
        state: XState,
        transitions: Sequence[XTransition],
    ) -> str:
        return str(Path(drive.root) / "domain.pddl")


@dataclass(frozen=True)
class FlashDriveMetadata(BaseDriveMetadata):
    iw_search: IWSearch | None


class FlashDrive(BaseDrive):
    def __init__(
        self,
        *args,
        iw_search: IWSearch | None = None,
        iw_options: Optional[Mapping[str, Any]] = None,
        attribute_getters: (
            dict[
                str,
                str
                | Callable[
                    [
                        "FlashDrive",
                        ExpandedStateSpaceEnv,
                        XState,
                        Sequence[XTransition],
                    ],
                    Any,
                ],
            ]
            | None
        ) = None,
        **kwargs,
    ) -> None:
        self.iw_search: IWSearch = iw_search
        self.iw_options = iw_options or dict()
        provided = attribute_getters or {}
        for key, value in provided.items():
            if isinstance(value, str):
                try:
                    provided[key] = getattr(attr_getters, value)
                except AttributeError:
                    try:
                        from jsonargparse.typing import import_object

                        value = import_object(value)
                        provided[key] = value
                    except Exception:
                        get_logger(__name__).error(
                            "Invalid attribute getter: %s for key '%s'", value, key
                        )
                        raise

            if not callable(value):
                raise TypeError(
                    f"Provided attribute getter for '{key}' must be a string or callable, got {type(value)}"
                )
            provided[key] = value
        default_getters = {
            "idx": attr_getters.idx,
            "object_count": attr_getters.object_count,
            "targets": attr_getters.targets,
            "distance_to_goal": attr_getters.distance_to_goal,
        }
        self.attr_getters = {**default_getters, **provided}
        if iw_search is not None:
            if isinstance(self.iw_search.expansion_strategy, RandomizedExpansion):
                get_logger(__name__).warning(
                    "Randomized expansion strategy in the IW search leads to a fixed expansion order in a stored FlashDrive."
                )
        super().__init__(*args, transform=self.target_idx_to_data_transform, **kwargs)

    def _metadata_misaligned(self, meta: dict) -> str:
        if not super()._metadata_misaligned(meta):
            if self.iw_search is not None and self.iw_search != meta["iw_search"]:
                return (
                    f"iw_search: given={self.iw_search} != loaded={meta['iw_search']}"
                )
        return ""

    @property
    def metadata(self) -> dict:
        return dict(
            **super().metadata,
            iw_search=self.iw_search,
            attribute_getters=[
                (k, v.__qualname__) for k, v in self.attr_getters.items()
            ],
        )

    def get_space(self):
        space = super().get_space()
        if self.iw_search is not None and self.iw_search.width > 0:
            space = IWStateSpace(self.iw_search, space, **self.iw_options)
        return space

    def _build(self) -> List[HeteroData]:
        env = self.env
        space: xmi.XStateSpace = env.active_instances[0]
        self.problem = space.problem
        encoder = self.encoder_factory(space.problem.domain)
        nr_states: int = len(space)
        logger = self._get_logger()
        logger.info(
            f"Building {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, space: {space})"
        )
        # Each data object represents one state
        batched_data: List[Union[HeteroData, Data]] = [None] * nr_states
        iterator = zip(
            space,
            env.get_applicable_transitions(space, instances=itertools.repeat(space)),
        )
        if self.show_progress:
            iterator = tqdm(
                iterator,
                total=nr_states,
                desc=f"{self.__class__.__name__}: PyG-encoding space",
                logger=get_logger(__name__),
            )
        for state, transitions in iterator:
            data = encoder.to_pyg_data(encoder.encode(state))
            data.reward, data.done = env.get_reward_and_done(
                transitions,
                instances=[space] * len(transitions),
            )
            for attr_name, getter in self.attr_getters.items():
                attr_data = getter(self, env, state, transitions)
                setattr(data, attr_name, attr_data)
            batched_data[state.index] = data
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})",
        )
        return batched_data

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
