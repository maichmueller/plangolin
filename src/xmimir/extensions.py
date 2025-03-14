from __future__ import annotations

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
def gather_objects(state: XState) -> set[Object]:
    return gather_objects(state.atoms())


@multimethod
def gather_objects(problem: XProblem) -> set[Object]:
    return set(problem.objects)


@multimethod
def gather_objects(atoms: Iterable[XAtom] | Generator) -> set[Object]:
    return set(chain.from_iterable(atom.objects for atom in atoms))


__all__ = ["parse", "gather_objects"]
