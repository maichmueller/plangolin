from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Generator, Iterable, Union

from multimethod import multimethod
from pymimir import Domain, Object, PDDLParser, Problem

from .wrappers import XAtom, XDomain, XProblem, XState


def parse(
    domain: Union[str, Path, Domain], problem: Union[str, Path, Problem]
) -> tuple[XDomain, XProblem]:
    """
    Parse the problem into a domain, problem, and repositories.
    """
    if isinstance(domain, Domain):
        domain = domain.get_filepath()
    if isinstance(problem, Problem):
        problem = problem.get_filepath()

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
