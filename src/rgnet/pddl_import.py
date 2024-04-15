import pathlib
from typing import Tuple, List

import pymimir as mi


def import_all_from(
    directory: pathlib.Path | str, domain_name: str = "domain"
) -> Tuple[mi.Domain, List[mi.Problem]]:
    """
    Import all pddl-problems and their domain from a directory.
    :param directory: The directory in which both problems and domain-file are located.
    :param domain_name: The exact file name (without suffix) of the domain-file
    :return: A tuple of domain and list of problems.
    """
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    assert directory.is_dir(), str(directory)
    domain_file = (directory / domain_name).with_suffix(".pddl")
    if not domain_file.exists():
        ValueError(f"Could not find domain file at {domain_file}")
    domain = mi.DomainParser(str(domain_file.absolute())).parse()
    problems = []
    for file in directory.glob("*.pddl"):
        if file.stem == domain_name:
            continue
        problems.append(mi.ProblemParser(str(file.absolute())).parse(domain))
    return domain, problems
