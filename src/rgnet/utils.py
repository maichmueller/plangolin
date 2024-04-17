import pathlib
import time
from datetime import timedelta

import networkx as nx
import torch


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
