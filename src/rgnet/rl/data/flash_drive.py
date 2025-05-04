import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

import torch
from torch_geometric.data import Data, HeteroData

import xmimir as xmi
from rgnet.logging_setup import tqdm
from rgnet.rl.envs import ExpandedStateSpaceEnv
from xmimir.iw import IWSearch, IWStateSpace, RandomizedExpansion

from .drive import BaseDrive, BaseDriveMetadata


@dataclass(frozen=True)
class FlashDriveMetadata(BaseDriveMetadata):
    iw_search: IWSearch | None


class FlashDrive(BaseDrive):
    def __init__(
        self,
        *args,
        iw_search: IWSearch | None = None,
        iw_options: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.iw_search: IWSearch = iw_search
        self.iw_options = iw_options or dict()
        if iw_search is not None:
            if isinstance(self.iw_search.expansion_strategy, RandomizedExpansion):
                logging.warning(
                    "Randomized expansion strategy in the IW search leads to a fixed expansion order in a stored FlashDrive."
                )
        super().__init__(*args, transform=self.target_idx_to_data_transform, **kwargs)

    def _metadata_misaligned(self, meta: FlashDriveMetadata) -> str:
        if not super()._metadata_misaligned(meta):
            if self.iw_search is not None and self.iw_search != meta.iw_search:
                return f"iw_search: given={self.iw_search} != loaded={meta.iw_search}"
        return ""

    @property
    def metadata(self) -> FlashDriveMetadata:
        return FlashDriveMetadata(
            **super().metadata.__dict__,
            iw_search=self.iw_search,
        )

    def _make_space(self):
        space = super()._make_space()
        if self.iw_search is not None and self.iw_search.width > 0:
            space = IWStateSpace(self.iw_search, space, **self.iw_options)
        return space

    def _build(
        self,
        env: ExpandedStateSpaceEnv,
    ) -> List[HeteroData]:
        out = env.traverse()[0]
        space: xmi.XStateSpace = out[env.keys.instance][0]
        encoder = self.encoder_factory(space.problem.domain)
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
