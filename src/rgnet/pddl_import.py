import logging
import pathlib
from typing import Tuple, List, Optional

import pymimir as mi


def _path_of_str(path: pathlib.Path | str) -> pathlib.Path:
    return path if isinstance(path, pathlib.Path) else pathlib.Path(path)


def import_all_from(
    directory: pathlib.Path | str, domain_name: str = "domain"
) -> Tuple[mi.Domain, List[mi.Problem]]:
    """
    Import all pddl-problems and their domain from a directory.
    :param directory: The directory in which both problems and domain-file are located.
    :param domain_name: The exact file name (without suffix) of the domain-file
    :return: A tuple of domain and list of problems.
    """
    directory = _path_of_str(directory)
    assert directory.is_dir(), str(directory)
    domain_file = (directory / domain_name).with_suffix(".pddl")
    if not domain_file.exists():
        ValueError(f"Could not find domain file at {domain_file}")
    domain = mi.DomainParser(str(domain_file.absolute())).parse()
    return domain, import_problems(directory, domain, domain_name=domain_name)


def import_problems(
    directory: pathlib.Path | str, domain: mi.Domain, domain_name: Optional[str] = None
) -> List[mi.Problem]:
    directory: pathlib.Path = _path_of_str(directory)
    assert directory.is_dir(), str(directory)
    domain_name = domain.name if domain_name is None else domain_name

    problems: List[mi.Problem] = []
    for file in directory.glob("*.pddl"):
        if file.stem == domain_name:
            continue
        try:
            problem = mi.ProblemParser(str(file.absolute())).parse(domain)
            assert problem is not None
            problems.append(problem)
        except (ValueError, AssertionError) as e:
            logging.warning(f"Skipped {file} while parsing problems: " + str(e))
    return problems
