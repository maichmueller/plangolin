import inspect
import re
from pathlib import Path
from test.fixtures import problem_setup

import pymimir as mi
import pytest

from rgnet.supervised.parse_serialized_dataset import DatasetParser


def test_validate_predicates():
    lines = """
        0 on
        1 ontable
        2 clear
        3 handempty
        4 holding"""
    predicate_lines = inspect.cleandoc(lines).splitlines()
    _, domain, problem = problem_setup("blocks", "small")
    parser = DatasetParser(domain, problem)
    parser.validate_predicates(predicate_lines)

    # create misconfiguration
    predicate_lines.append("5 invalid_predicate")
    with pytest.raises(ValueError):
        parser.validate_predicates(predicate_lines)

    predicate_lines = predicate_lines[:-1]
    predicate_lines[0] = predicate_lines[0].replace("0", "1")
    with pytest.raises(ValueError):
        parser.validate_predicates(predicate_lines)


def test_validate_objects():
    lines = """
        0 a
        1 b
    """
    object_lines = inspect.cleandoc(lines).splitlines()
    _, domain, problem = problem_setup("blocks", "small")
    parser = DatasetParser(domain, problem)
    parser.validate_objects(object_lines)

    # create misconfiguration
    object_lines.append("2 invalid_object")
    with pytest.raises(ValueError):
        parser.validate_predicates(object_lines)


def test_parse_labeled_state():
    lines = """0
        BEGIN_STATE
        1 0
        1 1
        2 0
        2 1
        3
        END_STATE"""
    labeled_state_lines = inspect.cleandoc(lines).splitlines()
    _, domain, problem = problem_setup("blocks", "small")
    parser = DatasetParser(domain, problem)
    label, state = parser.parse_labeled_state(labeled_state_lines)
    assert label == 0 and isinstance(state, mi.State)
    expected = {"clear(a)", "clear(b)", "ontable(a)", "ontable(b)", "handempty()"}
    assert set([atom.get_name() for atom in state.get_atoms()]) == expected


def test_parse():
    file = "test/pddl_instances/serialized/small_states.txt"
    _, domain, problem = problem_setup("blocks", "small")
    parser = DatasetParser(domain, problem)
    goals, labeled_states = parser.parse(Path(file))
    assert isinstance(goals, list) and isinstance(goals[0], mi.Atom)
    # find all matches of BEGIN_LABELED_STATE\n\d+ in file
    text = Path(file).read_text()
    matches = re.findall(r"BEGIN_LABELED_STATE\n(\d+)", text)
    expected = [int(label) for label in matches]
    assert [label for label, _ in labeled_states] == expected
