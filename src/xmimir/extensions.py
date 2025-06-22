from __future__ import annotations

from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Generator, Iterable, Union

from multimethod import multimethod
from pymimir import Domain, Object, PDDLParser, Problem

from .wrappers import XAtom, XDomain, XProblem, XState


def parse(
    domain: Union[str, Path, Domain, XDomain],
    problem: Union[str, Path, Problem, XProblem],
) -> tuple[XDomain, XProblem]:
    """
    Parse the problem into a domain, problem, and repositories.
    """
    match domain:
        case Domain():
            domain = domain.get_filepath()
        case XDomain():
            domain = domain.filepath
        case Path():
            domain = str(domain.absolute())
    match problem:
        case Problem():
            problem = problem.get_filepath()
        case XProblem():
            problem = problem.filepath
        case Path():
            problem = str(problem.absolute())

    parser = PDDLParser(domain, problem)
    return (
        XDomain(parser.get_domain()),
        XProblem(parser.get_problem(), parser.get_pddl_repositories()),
    )


@multimethod
def gather_objects(problem: XProblem) -> set[Object]:
    return set(problem.objects)


@multimethod
def gather_objects(atoms: Iterable[XAtom] | XState | Generator) -> set[Object]:
    return set(chain.from_iterable(atom.objects for atom in atoms))


def parse_atom_string(atom_str: str):
    # Basic parse for form: (predicate arg1 arg2 ...)
    atom_str = atom_str.strip()
    assert atom_str.startswith("(") and atom_str.endswith(")")
    parts = atom_str[1:-1].split()
    predicate = parts[0]
    objects = []
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            objects.append({k: v})
        else:
            objects.append(part)
    return predicate, objects


class StateType(Enum):
    DEFAULT = 0
    GOAL = 1
    INITIAL = 2
    DEADEND = 3


__all__ = ["parse", "gather_objects", "StateType", "parse_atom_string"]
