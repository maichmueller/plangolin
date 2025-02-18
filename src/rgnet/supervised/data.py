from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch_geometric.data import Batch, Data, HeteroData, InMemoryDataset

from rgnet.encoding import GraphEncoderBase
from xmimir import XProblem, XStateSpace


class MultiInstanceSupervisedSet(InMemoryDataset):

    @staticmethod
    def load_from(path: str):
        return MultiInstanceSupervisedSet(problems=None, state_encoder=None, root=path)

    def __init__(
        self,
        problems: List[XProblem] | None,
        state_encoder: GraphEncoderBase | None,
        max_expanded: Optional[int] = None,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        if (problems is None or state_encoder is None) and root is None:
            raise ValueError(
                "Neither list of problems nor path to stored dataset was provided"
            )
        self.problems: List[XProblem] = problems
        self.state_encoder = state_encoder
        self.max_expanded = max_expanded
        if problems is None and force_reload:
            raise ValueError("Tried to force reload problems but none where given.")
        # Will call self.process(...) if processed files do not exist or force_reload
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ["data.pt"]

    def download(self) -> None:
        pass

    def parse_problem(self, problem: XProblem) -> List[Data]:
        data_list = []
        space = XStateSpace.create(
            problem,
            max_num_states=self.max_expanded or 1_000_000,
        )
        encoder = self.state_encoder
        if space is None:  # None if more states than max_expanded
            logging.warning(f"Could not create state space for {problem}")
            return []
        for state in space:
            data: Data = encoder.to_pyg_data(encoder.encode(state))
            data.y = torch.tensor(space.goal_distance(state), dtype=torch.int64)
            data_list.append(data)
        return data_list

    def process(self) -> None:
        """
        Create a dataset containing a graph for each state of each problem.
        Use the StateSpace to compute the shortest path to a goal as label.
        For large state spaces this will probably take very long.
        """
        data_list = []
        if self.problems is None:
            raise ValueError("Tried to load existing dataset but could not find files.")

        for i, problem in enumerate(self.problems):
            data_list.extend(self.parse_problem(problem))
            if self.log:
                logging.info(f"Processed {i+1} / {len(self.problems)} problems")

        # Compute the avg value of data.y over all problems (excluding negative values).
        # torch.mean() wants to output floats, but we have to store ints.
        ys = torch.tensor([d.y for d in data_list])
        avg = ys[ys >= 0].mean(dtype=torch.float).long()
        # Replace dead-ends (value < 0) with the negative average.
        for data in data_list:
            if data.y < 0:
                data.y = -avg

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
