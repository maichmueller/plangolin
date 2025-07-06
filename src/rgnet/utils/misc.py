from __future__ import annotations

import copy
import functools
import hashlib
import logging
import os
import pathlib
import random
import time
from collections import deque
from datetime import timedelta
from functools import singledispatch
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Iterable, List, Reversible, Tuple, Union

import networkx as nx
import torch
from tensordict import NonTensorData, NonTensorStack
from torch_geometric.data import Batch

import xmimir as xmi


def get_colors(graph: nx.Graph):
    return sorted(set(graph[node]["color"].value for node in graph))


def get_device_cuda_if_possible(device_id: int | None = None) -> torch.device:
    if torch.cuda.is_available():
        if device_id is not None and device_id < torch.cuda.device_count():
            return torch.device(f"cuda:{device_id}")
        elif device_id is not None:
            raise ValueError(
                f"Requested CUDA device {device_id}, but only {torch.cuda.device_count()} are available."
            )
    return torch.device("cuda")


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


@singledispatch
def num_nodes_per_entry(
    data: Batch, node_type: str = None
) -> dict[str, torch.Tensor] | torch.Tensor:
    """
    Returns the number of nodes per graph-entry in the batch.

    :param data: The batch of graphs.
    :param node_type: The node type to count. If None, counts all node types.
    """
    if hasattr(data, "batch_dict"):
        batch_dict = data.batch_dict
        if node_type is None:
            return {
                ntype: _num_nodes_per_entry(batch[ntype])
                for ntype, batch in batch_dict.items()
            }
        else:
            return _num_nodes_per_entry(batch_dict[node_type])
    elif hasattr(data, "batch"):
        return _num_nodes_per_entry(data.batch)
    else:
        raise ValueError("No batch attribute found in data")


@num_nodes_per_entry.register
def _num_nodes_per_entry(batch: torch.Tensor) -> torch.Tensor:
    num_entries = batch.max() + 1
    out = torch.zeros(
        num_entries,
        dtype=torch.long,
        device=batch.device,
    )
    return out.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.long))


NonTensorWrapper = Union[NonTensorData, NonTensorStack]


def as_non_tensor_stack(sequence: Iterable) -> NonTensorStack:
    """
    Wrap every element of the list in a NonTensorData and stacks them into a
    NonTensorDataStack. We do not use torch.stack() in order to avoid getting
    NonTensorData returned, which is the case if all elements of the list are equal.
    """
    return NonTensorStack(*(NonTensorData(x) for x in sequence))


@singledispatch
def tolist(input_, **kwargs) -> List:
    return list(input_)


@tolist.register(list)
def _(input_: list, *, ensure_copy: bool = False, **kwargs) -> List:
    if ensure_copy:
        return input_.copy()
    return input_


@tolist.register(NonTensorStack)
@tolist.register(NonTensorData)
@tolist.register(torch.Tensor)
def _(input_: NonTensorWrapper, **kwargs) -> List:
    return input_.tolist()


@tolist.register(Batch)
def _(input_: Batch, **kwargs) -> List:
    return input_.to_data_list()


def as_forwarding_args(func, args, kwargs, defaults: dict[str, Any] | None = None):
    parent_sig = signature(func)
    kwargs = (defaults or {}) | kwargs
    bound_args = parent_sig.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args


def return_true(*args, **kwargs) -> bool:
    return True


def identity(x: Any) -> Any:
    """
    Identity function that returns the input unchanged.
    Useful as a default function argument.
    """
    return x


class ProbabilisticBuffer:
    """
    A fixed-size buffer that acts like a queue, but when full
    each new append has a 1/capacity chance to replace a random element.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self._buffer = deque()

    def append(self, item):
        """
        Add an item to the buffer.
        - If buffer not yet full: enqueue it.
        - If full: with probability 1/self.capacity, replace one random element.
        """
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            if random.random() < 1 / self.capacity:
                self._buffer.pop()
                self._buffer.append(item)

    def pop(self):
        """
        Remove and return the oldest item. Raises IndexError if empty.
        """
        return self._buffer.popleft()

    def __len__(self):
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def __repr__(self):
        return f"{self.__class__.__name__}(capacity={self.capacity}, buffer={list(self._buffer)})"
