import pathlib
from test.fixtures import problem_setup
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Batch, Dataset, HeteroData
from torch_geometric.loader import DataLoader

from rgnet.encoding import HeteroGraphEncoder
from rgnet.supervised.data import MultiInstanceSupervisedSet
from rgnet.utils.misc import broadcastable


def hetero_data_equal(data: HeteroData, expected: HeteroData):
    assert isinstance(data, HeteroData) and isinstance(expected, HeteroData)
    assert set(data.node_types) == set(expected.node_types)
    assert set(data.edge_types) == set(expected.edge_types)
    for key in data.node_types:
        dx = data[key].x
        ex = expected[key].x
        if dx.numel() == ex.numel() == 0:
            continue
        if not broadcastable(dx.shape, ex.shape):
            return False
        if not torch.allclose(dx, ex):
            return False
    for edge_type in data.edge_types:
        if not torch.equal(data[edge_type].edge_index, expected[edge_type].edge_index):
            return False
    return True


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


def create_dataset(size: str, root_dir: str | pathlib.Path, domain="blocks"):
    root_dir = root_dir if isinstance(root_dir, str) else str(root_dir.absolute())
    _, domain, problem = problem_setup(domain, size)
    encoder = HeteroGraphEncoder(domain)
    return MultiInstanceSupervisedSet(
        [problem], encoder, force_reload=True, root=root_dir
    )


def test_init(tmp_path):
    space, domain, problem = problem_setup("blocks", "small")
    encoder = HeteroGraphEncoder(domain)
    dataset = MultiInstanceSupervisedSet(
        [problem], encoder, root=str(tmp_path.absolute())
    )
    assert dataset.len() == len(space)
    assert all(
        data.y.dtype == torch.int64 and data.y.size() == torch.Size((1,))
        for data in dataset
    )


def test_collate(tmp_path):
    # Test that the reconstruction of individual data objects works correctly
    # across multiple instances, i.e.the collate and separate functions
    # Assumes that the dataset preserves the order of the saved data-list
    # Assumes that problem.get_states() returns the states in the same order
    small_space, domain, small_problem = problem_setup("blocks", "small")
    medium_space, _, medium_problem = problem_setup("blocks", "medium")
    encoder = HeteroGraphEncoder(domain)
    dataset = MultiInstanceSupervisedSet(
        [small_problem, medium_problem],
        encoder,
        root=str(tmp_path.absolute()),
    )
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

    dataset = create_dataset("medium", tmp_path)
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
    dataset = create_dataset("medium", str(tmp_path.absolute()))
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
