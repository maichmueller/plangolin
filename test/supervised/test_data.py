from typing import Callable, Optional

import pymimir as mi
import torch
from torch_geometric.data import Dataset

from rgnet.encoding import ColorGraphEncoder
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
