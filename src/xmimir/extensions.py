from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

from pymimir import Domain, PDDLParser, Problem

from .wrappers import XDomain, XProblem


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


__all__ = ["parse"]
