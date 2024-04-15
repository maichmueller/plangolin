import pymimir as mi
import torch

from rgnet.encoding import ColorGraphEncoder
from rgnet.supervised.data import MultiInstanceSupervisedSet


def test_init():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    encoder = ColorGraphEncoder(domain)
    dataset = MultiInstanceSupervisedSet([problem], encoder, force_reload=True)
    space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    assert dataset.len() == space.num_states()
    assert all(data.y.dtype == torch.float for data in dataset)
