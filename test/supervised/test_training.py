import pymimir as mi

from rgnet.encoding import ColorGraphEncoder
from rgnet.model import PureGNN
from rgnet.supervised.data import MultiInstanceSupervisedSet
from rgnet.supervised.training import training, evaluate


def test_training():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    model = training(MultiInstanceSupervisedSet([problem], ColorGraphEncoder(domain)))
    assert isinstance(model, PureGNN)


def test_evaluate():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    model = training(MultiInstanceSupervisedSet([problem], ColorGraphEncoder(domain)))
    evaluate(model, MultiInstanceSupervisedSet([problem], ColorGraphEncoder(domain)))
