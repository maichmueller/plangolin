import itertools
from argparse import ArgumentParser
from enum import auto

try:
    from enum import StrEnum  # Available in Python 3.11+
except ImportError:
    from strenum import StrEnum  # Backport for Python < 3.11
from typing import Optional

import torch
import torch_geometric.nn.aggr

from rgnet.encoding import HeteroGraphEncoder
from rgnet.rl.data_layout import InputData
from rgnet.rl.embedding import EmbeddingModule, build_hetero_embedding_and_gnn
from rgnet.utils.misc import tolist


def one_hot_embedding(
    all_states, hidden_size: Optional[int] = None, device=None
) -> EmbeddingModule:
    hidden_size = hidden_size or len(all_states)
    num_states = len(all_states)
    device = device or torch.device("cpu")
    embedding: torch.Tensor
    if len(all_states) == hidden_size:
        embedding = torch.eye(
            hidden_size, hidden_size, dtype=torch.float, device=device
        )
    elif num_states > hidden_size:
        raise ValueError("Number of states must be less than or equal to hidden size")
    else:
        embedding = torch.nn.functional.one_hot(
            torch.arange(num_states, device=device), hidden_size
        ).float()

    class Embedding(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.lookup = {s: e for s, e in zip(all_states, embedding)}
            self.hidden_size = hidden_size
            self.device = device

        def forward(self, states_):
            states_ = tolist(states_)
            return torch.stack([self.lookup[s] for s in states_])

    return Embedding()


class Parameter(StrEnum):
    embedding_type = auto()
    gnn_hidden_size = auto()
    gnn_num_layer = auto()
    gnn_aggr = auto()


def from_parser_args(
    parser_args, device: torch.device, data_resolver: InputData
) -> EmbeddingModule:
    if getattr(parser_args, Parameter.embedding_type) == "one_hot":
        return one_hot_embedding(
            all_states=list(
                itertools.chain.from_iterable(
                    [s.get_states() for s in data_resolver.spaces]
                )
            ),
            device=device,
        )

    encoder = HeteroGraphEncoder(domain=data_resolver.domain)
    aggr = getattr(parser_args, Parameter.gnn_aggr)
    if aggr == "softmax":
        aggr = torch_geometric.nn.aggr.SoftmaxAggregation()

    return build_hetero_embedding_and_gnn(
        encoder=encoder,
        hidden_size=getattr(parser_args, Parameter.gnn_hidden_size),
        num_layer=getattr(parser_args, Parameter.gnn_num_layer),
        aggr=aggr,
        device=device,
    )


def add_parser_args(parent_parser: ArgumentParser):
    parser = parent_parser.add_argument_group("Embedding")
    parser.add_argument(
        f"--{Parameter.embedding_type.value}",
        choices=["one_hot", "gnn"],
        required=True,
        default="gnn",
        help="Type of embedding to use (default: gnn)",
    )
    parser.add_argument(
        f"--{Parameter.gnn_hidden_size.value}",
        type=int,
        required=False,
        default=32,
        help="Hidden size for the GNN (default: 32).",
    )
    parser.add_argument(
        f"--{Parameter.gnn_num_layer.value}",
        type=int,
        required=False,
        default=3,
        help="Number of layers for the GNN (default: 3)",
    )
    parser.add_argument(
        f"--{Parameter.gnn_aggr.value}",
        choices=["mean", "max", "sum", "softmax"],
        required=False,
        default="sum",
        help="Type of aggregation for the GNN (default: sum)",
    )
    return parent_parser
