from typing import Union, List, Tuple, Optional, Callable

from pymimir import Problem, StateSpace, GroundedSuccessorGenerator
from torch_geometric.data import InMemoryDataset, Data

from rgnet.encoding import ColorGraphEncoder


class MultiInstanceSupervisedSet(InMemoryDataset):

    def __init__(
        self,
        problems: List[Problem],
        state_encoder: ColorGraphEncoder,
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
        return []  # TODO empty list triggers download every time

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
        for problem in self.problems:
            space = StateSpace.new(problem, GroundedSuccessorGenerator(problem))
            for state in space.get_states():
                data: Data = self.encoder.encoding_to_pyg_data(state)
                data.y = float(space.get_distance_to_goal_state(state))
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
