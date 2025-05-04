from __future__ import annotations

import copy
import functools
import hashlib
import logging
import os
import pathlib
import time
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import Any, Callable, Iterable, List, Reversible, Tuple

import networkx as nx
import torch
import torch_geometric

import xmimir as xmi
from xmimir import StateType


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


def env_aware_cpu_count(ignore_slurm: bool = False) -> int:
    """
    Returns the number of CPUs available on the machine.
    :return: Number of CPUs
    """
    return int(
        os.cpu_count()
        if ignore_slurm
        else os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())
    )


def persistent_hash(data: Iterable[Any], sep: str = ",") -> str:
    """
    When saving to disk under a hash folder name, you need to use a stable hash.
    The builtin function `hash` is deterministic only WITHIN the same execution of the interpreter.
    It is not guaranteed to be the same across different runs/platforms/python-versions.
    """
    sha1 = hashlib.sha1(str.encode(sep.join(str(d) for d in data)))
    metadata_hash_hexa = sha1.hexdigest()
    return str(metadata_hash_hexa)


@cache
def mdp_graph_as_pyg_data(nx_state_space_graph: nx.DiGraph):
    """
    Convert the networkx graph into a directed pytorch_geometric graph.
    The reward for each transition is stored in edge_attr[:, 0].
    The transition probabilities are stored in edge_attr[:, 1].
    The node features are stored as usual in graph.x.
    The first dimension is the node value (starting with 0).
    The second node feature dimension is one, if the node is a goal state.
    """
    pyg_graph = torch_geometric.utils.from_networkx(
        nx_state_space_graph, group_edge_attrs=["reward", "probs", "idx"]
    )
    transition_indices = pyg_graph.edge_attr[:, 2]
    expected_transition_indices = torch.arange(transition_indices.max().item() + 1)
    if (transition_indices.int() != expected_transition_indices).any():
        # we have to maintain the order of the edges as they are returned by a traversal of the state space.
        graph_clone = pyg_graph.clone()
        sorted_transition_indices = torch.argsort(graph_clone.edge_attr[:, 2])
        pyg_graph.edge_index = graph_clone.edge_index[:, sorted_transition_indices]
        pyg_graph.edge_attr = graph_clone.edge_attr[sorted_transition_indices, :]
    transition_indices = pyg_graph.edge_attr[:, 2]
    assert (transition_indices.int() == expected_transition_indices).all()
    is_goal_state = [False] * pyg_graph.num_nodes
    # inf as default to trigger errors if logic did not hold
    goal_reward = [float("inf")] * pyg_graph.num_nodes
    for i, (node, attr) in enumerate(nx_state_space_graph.nodes.data()):
        if attr["ntype"] == StateType.GOAL:
            is_goal_state[i] = True
            _, _, goal_reward[i] = next(
                iter(nx_state_space_graph.out_edges(node, data="reward"))
            )

    pyg_graph.goals = torch.tensor(
        is_goal_state,
        dtype=torch.bool,
    )
    # goal states have the value of their reward (typically 0, but could be arbitrary);
    # the rest is initialized to 0.
    pyg_graph.x = torch.where(
        pyg_graph.goals,
        torch.tensor(goal_reward, dtype=torch.float),
        torch.zeros((pyg_graph.num_nodes,)),
    )
    if hasattr(nx_state_space_graph.graph, "gamma"):
        pyg_graph.gamma = nx_state_space_graph.graph["gamma"]
    return pyg_graph


@functools.wraps
def copy_return(func):
    """
    Decorator to copy the return value of a function.
    This is useful for functions that return a tensor, as it ensures that the tensor is copied to the CPU.
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if hasattr(result, "clone") and callable(result.clone):
            return result.clone()
        else:
            copy.copy(result)
        return result

    return wrapper
