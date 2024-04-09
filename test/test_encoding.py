import pymimir as mi
from rgnet.encoding import ColorGraphEncoder


def test_encoding():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    state_space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    state = state_space.get_initial_state()
    graph = ColorGraphEncoder(problem).encode(state)
    x = 34
