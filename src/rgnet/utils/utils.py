from __future__ import annotations

import logging
import pathlib
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, List, Reversible, Tuple

import networkx as nx
import torch

import xmimir as xmi


def get_colors(graph: nx.Graph):
    return sorted(set(graph[node]["color"].value for node in graph))


def get_device_cuda_if_possible() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def broadcastable(shape1: Reversible[int], shape2: Reversible[int]):
    for a, b in zip(reversed(shape1), reversed(shape2)):
        if a != b and a != 1 and b != 1:
            return False
    return True


def import_all_from(
    directory: Path | str, domain_filename: str = "domain"
) -> Tuple[xmi.XDomain, List[xmi.XProblem]]:
    """
    Import all pddl-problems and their domain from a directory.
    :param directory: The directory in which both problems and domain-file are located.
    :param domain_filename: The exact file name (without suffix) of the domain-file
    :return: A tuple of domain and list of problems.
    """
    directory = Path(directory)
    assert directory.is_dir(), str(directory)
    domain_file = (directory / domain_filename).with_suffix(".pddl")
    if not domain_file.exists():
        ValueError(f"Could not find domain file at {domain_file}")
    return import_problems(directory, domain_file)


def import_problems(
    directory: pathlib.Path | str, domain_filepath: Path
) -> Tuple[xmi.XDomain, List[xmi.XProblem]]:
    directory: pathlib.Path = Path(directory)
    assert directory.is_dir(), str(directory)

    problems: List[xmi.XProblem] = []
    files = list(filter(lambda fp: fp != domain_filepath, directory.glob("*.pddl")))
    if not files:
        raise ValueError(f"No problems found in {directory}")
    for file in files:
        try:
            domain, problem = xmi.parse(
                str(domain_filepath.absolute()), str(file.absolute())
            )
            assert problem is not None
            problems.append(problem)
        except (ValueError, AssertionError) as e:
            logging.warning(f"Skipped {file} while parsing problems: " + str(e))

    return domain, problems


class KeyAwareDefaultDict(dict):
    def __init__(self, default_factory: Callable[[Any], Any], *args, **kwargs):
        if not callable(default_factory):
            raise TypeError("default_factory must be a callable")
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        # Generate the default value using the key
        value = self.default_factory(key)
        self[key] = value  # Store the value in the dictionary
        return value
