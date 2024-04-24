import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from pymimir import GroundedSuccessorGenerator, Problem, StateSpace
from torch_geometric.data import Batch, Data, HeteroData, InMemoryDataset

from rgnet.encoding import StateEncoderBase


class MultiInstanceSupervisedSet(InMemoryDataset):
    def __init__(
        self,
        problems: List[Problem],
        state_encoder: StateEncoderBase,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        self.problems = problems
        self.encoder = state_encoder
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [p.name for p in self.problems]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ["data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        """
        Create a dataset containing a graph for each state of each problem.
        Use the StateSpace to compute the shortest path to a goal as label.
        For large state spaces this will probably take very long.
        """
        data_list = []
        for i, problem in enumerate(self.problems):
            space = StateSpace.new(problem, GroundedSuccessorGenerator(problem))
            for state in space.get_states():
                data: Data = self.encoder.to_pyg_data(self.encoder.encode(state))
                data.y = torch.tensor(
                    space.get_distance_to_goal_state(state), dtype=torch.int64
                )
                data_list.append(data)
            if self.log:
                logging.info(f"Processed {i+1} / {len(self.problems)} problems")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def get_label_distribution(self) -> torch.Tensor:
        """
        Return a tensor of label frequencies. where the i-th element
         is the frequency of i in the dataset.
        :return: torch.Tensor of frequencies of shape Size([max(self.y) + 1])
        """
        return torch.bincount(self.y.int())

    def __getattr__(self, key: str) -> Any:
        """InMemoryDataset forgot the poor HeteroData objects, logic is equivalent."""
        data = self.__dict__.get("_data")
        if isinstance(data, HeteroData) and key in data:
            if self._indices is None and data.__inc__(key, data[key]) == 0:
                return data[key]
            else:
                data_list = [self.get(i) for i in self.indices()]
                return Batch.from_data_list(data_list)[key]
        else:
            return super().__getattr__(key)
