from test.fixtures import *

from xmimir import *
from xmimir.wrappers import MimirWrapper


def test_predicates(medium_blocks):
    space, domain, problem = medium_blocks
    predicates = domain.predicates()
    other_domain, other_problem = parse(domain, problem)
    other_predicates = other_domain.predicates()
    # == checks for base equality which means pymimir checks if the two predicates are the very same object in memory
    assert all(all(p != q for p in predicates) for q in other_predicates)
    # semantic_eq_sequences checks if the two predicates are the same in terms of content
    assert MimirWrapper.semantic_eq_sequences(
        predicates, other_predicates, ordered=False
    )


def test_literals(medium_blocks):
    space, domain, problem = medium_blocks
    other_domain, other_problem = parse(domain, problem)
    literals = tuple(problem.initial_literals())
    other_literals = tuple(other_problem.initial_literals())

    assert all(all(p != q for p in literals) for q in other_literals)
    assert MimirWrapper.semantic_eq_sequences(literals, other_literals, ordered=False)

    goals = tuple(problem.goal())
    other_goals = tuple(other_problem.goal())

    assert all(all(p != q for p in goals) for q in other_goals)
    assert MimirWrapper.semantic_eq_sequences(goals, other_goals, ordered=False)


def test_atoms(small_blocks, medium_blocks):
    space, domain, problem = medium_blocks
    other_domain, other_problem = parse(domain, problem)
    atoms = tuple(problem.initial_atoms())
    other_atoms = tuple(other_problem.initial_atoms())

    assert all(all(p != q for p in atoms) for q in other_atoms)
    assert MimirWrapper.semantic_eq_sequences(atoms, other_atoms, ordered=False)

    space2 = XStateSpace(problem)
    states = tuple(space)
    states2 = tuple(space2)

    assert all(all(p != q for p in states) for q in states2)
    assert MimirWrapper.semantic_eq_sequences(states, states2, ordered=False)

    small_space, small_domain, small_problem = small_blocks

    states3 = tuple(small_space)

    assert all(all(p != q for p in states) for q in states3)
    assert not MimirWrapper.semantic_eq_sequences(states, states3, ordered=False)
    assert not MimirWrapper.semantic_eq_sequences(states2, states3, ordered=False)
