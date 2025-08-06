from test.fixtures import make_fresh_flashdrive, problem_setup
from test.test_utils import hetero_data_equal
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Batch, Dataset, HeteroData
from torch_geometric.loader import DataLoader

from rgnet.rl.data.flash_drive import attr_getters


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


def test_collate(tmp_path):
    # Test that the reconstruction of individual data objects works correctly
    # across multiple instances, i.e.the collate and separate functions
    # Assumes that the dataset preserves the order of the saved data-list
    # Assumes that problem.get_states() returns the states in the same order
    domain_name, problem_name = "blocks", "small"
    dataset = make_fresh_flashdrive(
        tmp_path,
        domain_name,
        problem_name,
        force_reload=False,
        attribute_getters={"y": attr_getters.distance_to_goal},
    )
    small_space, domain, small_problem = problem_setup("blocks", "small")
    medium_space, _, medium_problem = problem_setup("blocks", "medium")
    encoder = dataset.encoder_factory(domain)
    manual_data: List[HeteroData] = []
    for problem, space in zip(
        [small_problem, medium_problem], [small_space, medium_space]
    ):
        for state in space:
            data = encoder.to_pyg_data(encoder.encode(state))
            data.y = torch.tensor(space.goal_distance(state), dtype=torch.int64)

            manual_data.append(data)
    data: HeteroData
    expected: HeteroData
    for i, (data, expected) in enumerate(zip(dataset, manual_data)):
        assert hetero_data_equal(data, expected)
        assert torch.all(data.y == expected.y)


def test_batched(tmp_path):
    # Tests Batch.from_list, depends on success of test_collate
    domain_name, problem_name = "blocks", "medium"
    dataset = make_fresh_flashdrive(
        tmp_path,
        domain_name,
        problem_name,
        force_reload=False,
        attribute_getters={"y": attr_getters.distance_to_goal},
    )
    batched: Batch = Batch.from_data_list(dataset)

    for i, expected in enumerate(dataset):
        data = batched.get_example(i)
        assert hetero_data_equal(data, expected)
        assert torch.all(data.y == expected.y)

    sliced = dataset[dataset.y == 0]
    sliced_batched = Batch.from_data_list(sliced)
    for i, expected in enumerate(sliced):
        data = sliced_batched.get_example(i)
        assert hetero_data_equal(data, expected)
        assert torch.all(data.y == expected.y)


def test_loader(tmp_path):
    # Test that data loaders correctly batch data, depends on success of test_collate
    domain_name, problem_name = "blocks", "medium"
    dataset = make_fresh_flashdrive(
        tmp_path, domain_name, problem_name, force_reload=False
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
    )
    for i, batch in enumerate(loader):
        assert isinstance(batch, Batch)
        data0: HeteroData = batch.get_example(0)
        assert hetero_data_equal(data0, dataset[i * 2])
        if batch.batch_size == 2:
            data1 = batch.get_example(1)
            assert hetero_data_equal(data1, dataset[i * 2 + 1])
