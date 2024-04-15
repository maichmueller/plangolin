from typing import Optional

import networkx as nx
import torch


def get_colors(graph: nx.Graph):
    return sorted(set(graph[node]["color"].value for node in graph))


def require_not_none(obj, message: Optional[str] = None):
    if obj is None:
        raise ValueError(
            message if message is not None else f"Required obj to be not None"
        )


def get_device_cuda_if_possible():
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
