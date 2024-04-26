from test.encoding.test_color_encoder import problem_setup
from typing import Callable, List, Optional

import pymimir as mi
import torch
from torch_geometric.data import Dataset, HeteroData

from rgnet.encoding import ColorGraphEncoder, HeteroEncoding
from rgnet.supervised.data import MultiInstanceSupervisedSet


class EmptyDataset(Dataset):

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")


def test_init():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/small.pddl").parse(domain)
    encoder = ColorGraphEncoder(domain)
    dataset = MultiInstanceSupervisedSet([problem], encoder, force_reload=True)
    space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    assert dataset.len() == space.num_states()
    assert all(
        data.y.dtype == torch.int64 and data.y.size() == torch.Size((1,))
        for data in dataset
    )


def test_collate():
    # Test that the reconstruction of individual data objects works correctly
    # across multiple instances, i.e.the collate and separate functions
    # Assumes that the dataset preserves the order of the saved data-list
    # Assumes that problem.get_states() returns the states in the same order
    small_space, domain, small_problem = problem_setup("blocks", "small")
    medium_space, _, medium_problem = problem_setup("blocks", "medium")
    encoder = HeteroEncoding(domain, 2)
    dataset = MultiInstanceSupervisedSet(
        [small_problem, medium_problem], encoder, force_reload=True
    )
    manual_data: List[HeteroData] = []
    for space in [small_space, medium_space]:
        for state in space.get_states():
            data = encoder.to_pyg_data(encoder.encode(state))
            data.y = torch.tensor(
                space.get_distance_to_goal_state(state), dtype=torch.int64
            )
            manual_data.append(data)
    data: HeteroData
    expected: HeteroData
    for i, (data, expected) in enumerate(zip(dataset, manual_data)):
        compare_hetero_stores(data, expected)
        assert torch.all(data.y == expected.y)


def compare_hetero_stores(data, expected):
    assert isinstance(data, HeteroData) and isinstance(expected, HeteroData)
    assert set(data.node_types) == set(expected.node_types)
    assert set(data.edge_types) == set(expected.edge_types)
    for key in data.node_types:
        dx = data[key].x
        ex = expected[key].x
        if dx.numel() == ex.numel() == 0:
            continue
        assert torch.allclose(dx, ex)
    for edge_type in data.edge_types:
        assert torch.all(data[edge_type].edge_index == expected[edge_type].edge_index)
