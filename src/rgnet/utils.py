import pathlib

import networkx as nx
import torch


def get_colors(graph: nx.Graph):
    return sorted(set(graph[node]["color"].value for node in graph))


def get_device_cuda_if_possible() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path_of_str(path: pathlib.Path | str) -> pathlib.Path:
    return path if isinstance(path, pathlib.Path) else pathlib.Path(path)
