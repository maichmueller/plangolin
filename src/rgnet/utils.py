import time
from datetime import timedelta

import networkx as nx
import torch

import logging
import pathlib
from typing import Tuple, List, Optional

import pymimir as mi


def get_colors(graph: nx.Graph):
    return sorted(set(graph[node]["color"].value for node in graph))


def get_device_cuda_if_possible() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path_of_str(path: pathlib.Path | str) -> pathlib.Path:
    return path if isinstance(path, pathlib.Path) else pathlib.Path(path)


def time_delta_now(previous: float) -> str:
    return ftime(time.time() - previous)


def ftime(seconds: float) -> str:
    delta = (
        timedelta(seconds=int(seconds)) if seconds >= 60 else timedelta(seconds=seconds)
    )

    if delta.days > 0:
        return str(delta) + "d"
    if delta.seconds >= 3600:
        return str(delta) + "h"
    if delta.seconds >= 60:
        return str(delta)[2:] + "m"
    if delta.seconds >= 1:
        return str(delta.seconds) + "s"
    if delta.microseconds > 1000:
        return str(delta.microseconds // 1000) + "ms"
    return str(delta.microseconds) + "us"


def import_all_from(
    directory: pathlib.Path | str, domain_name: str = "domain"
) -> Tuple[mi.Domain, List[mi.Problem]]:
    """
    Import all pddl-problems and their domain from a directory.
    :param directory: The directory in which both problems and domain-file are located.
    :param domain_name: The exact file name (without suffix) of the domain-file
    :return: A tuple of domain and list of problems.
    """
    directory = path_of_str(directory)
    assert directory.is_dir(), str(directory)
    domain_file = (directory / domain_name).with_suffix(".pddl")
    if not domain_file.exists():
        ValueError(f"Could not find domain file at {domain_file}")
    domain = mi.DomainParser(str(domain_file.absolute())).parse()
    return domain, import_problems(directory, domain, domain_name=domain_name)


def import_problems(
    directory: pathlib.Path | str, domain: mi.Domain, domain_name: Optional[str] = None
) -> List[mi.Problem]:
    directory: pathlib.Path = path_of_str(directory)
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
