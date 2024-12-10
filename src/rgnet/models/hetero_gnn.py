from typing import Callable, Dict, Optional, Union

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.typing import Adj

from rgnet.models.hetero_message_passing import FanInMP, FanOutMP


def mlp(in_size: int, hidden_size: int, out_size: int, **kwargs):
    return pyg.nn.MLP(
        [in_size, hidden_size, out_size], norm=None, dropout=0.0, **kwargs
    )


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        activation: Union[str, Callable, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.mlp = mlp(hidden_size, hidden_size, hidden_size, act=activation)

    def forward(self, input_tensor: torch.Tensor):
        return input_tensor + self.mlp(input_tensor)


class HeteroGNN(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
        pool: Optional[Union[str, Callable[[Tensor, Tensor], Tensor]]] = None,
        split_embeddings: bool = False,
        activation: Union[str, Callable, None] = None,
    ):
        """
        :param hidden_size: The size of object embeddings.
        :param num_layer: Total number of message exchange iterations.
        :param aggr: Aggregation function to be used for message passing.
        :param obj_type_id: The type identifier of objects in the x_dict.
        :param arity_dict: A dictionary mapping predicate names to their arity.
        :param activation: The activation function for all MLPs
            (Default reLU).
        Creates one MLP for each predicate.
        Note that predicates as well as goal-predicates are meant.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layer: int = num_layer
        self.obj_type_id: str = obj_type_id
        self.split_embeddings: bool = split_embeddings
        if pool is not None and isinstance(pool, str) and pool:
            if pool == "add":
                pool = pyg.nn.global_add_pool
            elif pool == "mean":
                pool = pyg.nn.global_mean_pool
            elif pool == "max":
                pool = pyg.nn.global_max_pool
            else:
                raise ValueError(
                    f"Unknown pooling function: {pool}. Choose from [add, mean, max]."
                )
        self.pool = pool
        if aggr == "softmax":
            aggr = SoftmaxAggregation()

        mlp_dict = {
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            pred: ResidualBlock(hidden_size * arity, activation=activation)
            for pred, arity in arity_dict.items()
            if arity > 0
        }

        self.obj_to_atom = FanOutMP(mlp_dict, src_name=obj_type_id)

        self.obj_update = mlp(
            in_size=2 * hidden_size,
            hidden_size=2 * hidden_size,
            out_size=hidden_size,
            act=activation,
        )
        # Messages from atoms flow to objects
        self.atom_to_obj = FanInMP(
            hidden_size=hidden_size,
            dst_name=obj_type_id,
            aggr=aggr,
        )

    def encoding_layer(self, x_dict: Dict[str, Tensor]):
        # Resize everything by the hidden_size
        # embedding of objects = hidden_size
        # embedding of atoms = arity of predicate * hidden_size
        for k, v in x_dict.items():
            assert v.dim() == 2
            x_dict[k] = torch.zeros(
                v.shape[0], v.shape[1] * self.hidden_size, device=v.device
            )
        return x_dict

    def layer(self, x_dict, edge_index_dict):
        # Groups object embeddings that are part of an atom and
        # applies predicate-specific MLP based on the edge type.
        out = self.obj_to_atom(x_dict, edge_index_dict)
        x_dict.update(out)  # update atom embeddings
        # Distribute the atom embeddings back to the corresponding objects.
        out = self.atom_to_obj(x_dict, edge_index_dict)
        # Update the object embeddings using a shared update-MLP.
        obj_emb = torch.cat([x_dict[self.obj_type_id], out[self.obj_type_id]], dim=1)
        obj_emb = self.obj_update(obj_emb)
        x_dict[self.obj_type_id] = obj_emb

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):
        # Filter out dummies
        x_dict = {k: v for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.numel() != 0}

        x_dict = self.encoding_layer(x_dict)  # Resize everything by the hidden_size

        for _ in range(self.num_layer):
            self.layer(x_dict, edge_index_dict)

        obj_emb = x_dict[self.obj_type_id]
        batch = (
            batch_dict[self.obj_type_id]
            if batch_dict is not None
            else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=obj_emb.device)
        )
        if self.pool is not None:
            # Aggregate all object embeddings into one aggregated embedding
            return self.pool(obj_emb, batch)  # shape [hidden, 1]
        if self.split_embeddings:
            return self.split_object_embeddings(obj_emb, batch)
        else:
            return obj_emb

    @staticmethod
    def split_object_embeddings(object_embeddings: torch.Tensor, batch_indices: Tensor):
        """
        Splits object embeddings by batch indices into embedding tensors for each batch element.

        Example
        -------
        Batch object embeddings of shape (N, D) are split into a list of tensors of shape (N_i, D) where N_i is the
        number of objects in batch element i and hence sum_i N_i = N.
        """
        # compute sizes of each batch
        unique_batches, batch_sizes = torch.unique(batch_indices, return_counts=True)
        # split embeddings by batch sizes
        grouped_embeddings = torch.split(object_embeddings, batch_sizes.tolist())
        return grouped_embeddings


class ValueHeteroGNN(HeteroGNN):

    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        obj_type_id: str,
        arity_dict: Dict[str, int],
        activation: Union[str, Callable, None] = None,
    ):
        super().__init__(
            hidden_size,
            num_layer,
            aggr,
            obj_type_id,
            arity_dict,
            activation=activation,
        )
        self.readout = mlp(hidden_size, 2 * hidden_size, 1, act=activation)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):
        aggr = super().forward(x_dict, edge_index_dict, batch_dict)
        return self.readout(aggr).view(-1)
