import itertools
from typing import Callable, Iterable, NamedTuple, Sequence

import torch
import torch_geometric.nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver

from rgnet.utils.batching import batched_permutations
from rgnet.utils.misc import num_nodes_per_entry, tolist
from xmimir import XCategory, XDomain, XPredicate
from xmimir.wrappers import atom_str_template

from .hetero_gnn import simple_mlp
from .patched_module_dict import PatchedModuleDict
from .residual import ResidualModule


class OutputInfo(NamedTuple):
    state_index: int
    atom: str


class AtomValuator(torch.nn.Module):
    def __init__(
        self,
        predicates: Iterable[XPredicate] | XDomain,
        predicate_module_factory: (
            Callable[[XPredicate, int], torch.nn.Module] | None
        ) = None,
        pooling: str = "sum",
        activation: str = "mish",
        feature_size: int = 64,
        assert_output: bool = False,
    ):
        super().__init__()
        if isinstance(predicates, XDomain):
            predicates = predicates.predicates(XCategory.fluent, XCategory.derived)
        self.predicates_dict = {p.name: p for p in predicates}
        self.arity_dict: dict[str, int] = {
            p.name: p.arity for p in self.predicates_dict.values()
        }
        self.arity_to_predicates: dict[int, list[str]] = {
            arity: sorted(
                predicate
                for predicate, pred_arity in self.arity_dict.items()
                if arity == pred_arity
            )
            for arity in sorted(self.arity_dict.values())
        }
        assert sum(len(ps) for ps in self.arity_to_predicates.values()) == len(
            self.predicates_dict
        )
        self.predicate_to_index = {
            p.name: p.base.get_index() for p in self.predicates_dict.values()
        }
        self.max_arity = max(self.arity_dict.values())
        self.valuator_by_predicate = PatchedModuleDict(
            {
                # One Module per predicate
                # For a predicate p(o_1,...,o_k) the corresponding Module gets k object
                # embeddings as input and generates k outputs, one for each object.
                pred: (
                    torch.nn.Sequential(
                        ResidualModule(
                            simple_mlp(
                                # arity + 1 to allow for entering the aggregated state information
                                *[feature_size * (arity + 1)] * 3,
                                activation=activation,
                            )
                        ),
                        activation_resolver(activation),
                        torch.nn.Linear(feature_size * (arity + 1), 1),
                    )
                    if predicate_module_factory is None
                    else predicate_module_factory(
                        self.predicates_dict[pred], feature_size
                    )
                )
                for pred, arity in self.arity_dict.items()
            }
        )
        match pooling:
            case "sum":
                self.pooling = torch_geometric.nn.global_add_pool
            case "add":
                self.pooling = torch_geometric.nn.global_add_pool
            case "mean":
                self.pooling = torch_geometric.nn.global_mean_pool
            case "max":
                self.pooling = torch_geometric.nn.global_max_pool
            case _:
                raise ValueError(
                    f"Unknown state pooling function: {pooling}. "
                    f"Choose from [sum, add, max, mean]."
                )
        self.feature_size = feature_size
        self.assert_output = assert_output
        self._streams: dict[str, torch.cuda.Stream] = dict()

    @property
    def device(self):
        return next(self.parameters()).device

    def stream_for(self, predicate: str) -> torch.cuda.Stream | None:
        if self.device.type == "cpu":
            return None
        if not self._streams and self.device.type == "cuda":
            self._streams = {
                pred: torch.cuda.Stream(self.device) for pred in self.predicates_dict
            }
        return self._streams[predicate]

    def forward(
        self,
        embeddings: Tensor,
        batch_info: Tensor,
        object_names: list[list[str]],
        with_info: bool | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, list[OutputInfo]]]:
        """
        Batch is a PyG Batch object containing the graphs of batch_size many states.

        We compute the embeddings of the objects in the states and split up the tensor into state-respective
        object-embedding pieces, i.e. each state has its own object-embedding 2D-tensor where dim-0 is the object
        dimension and dim-1 the feature dimension.
        We then compute all possible permutations (with replacement) of object-embeddings of size `arity(predicate)`
        in each state and batch them together again.

        The predicate-specific MLP then performs a readout of the atom values.

        :param embeddings: The object embeddings for the batch of states. We assume that the
            embeddings are already computed and passed as input.
            The shape of the embeddings should be:
                (batch_size * avg_object_count_per_state, feature_size).
        :param batch_info: A tensor indicating for each entry in the batch which state it belongs to.
        :param object_names: A list containing the list of object names in each state.
        :param with_info: If True, returns additional information about the output for verification.
        """
        if with_info is None:
            with_info = self.assert_output
        num_objects = num_nodes_per_entry(batch_info)
        object_emb_per_state: tuple[Tensor, ...] = torch.split(
            embeddings, tolist(num_objects)
        )
        predicate_out = dict()
        output_info: dict[str, list[OutputInfo]] = dict()
        pooled_emb_batch = self.pooling(embeddings, batch_info)
        for arity in self.arity_to_predicates.keys():
            if arity == 0:
                # there are no objects in the predicate,
                # so we only compute the atom's value from the state aggregation alone
                for predicate in self.arity_to_predicates[arity]:
                    predicate_out[predicate], output_info[predicate] = (
                        self._forward_predicate(
                            predicate,
                            pooled_emb_batch,
                            range(len(object_emb_per_state)),
                            itertools.repeat([]),
                            with_info=with_info,
                        )
                    )
            else:
                flattened_permuted_emb = []
                objects_per_atom = []
                state_association = []
                repeated_state_aggr_emb = []
                for state_index, (
                    state_object_names,
                    state_object_embs,
                    state_aggr_emb,
                ) in enumerate(
                    zip(object_names, object_emb_per_state, pooled_emb_batch)
                ):
                    for object_permut_batch in batched_permutations(
                        state_object_embs,
                        arity,
                        batch_size=1,
                        with_replacement=True,  # allow repeated objects in atoms as args, e.g. on(a a), (= b b), etc.
                    ):
                        nr_perms = len(object_permut_batch.permutations)
                        state_association.extend([state_index] * nr_perms)
                        repeated_state_aggr_emb.extend(
                            [state_aggr_emb.view(1, -1)] * nr_perms
                        )
                        for (
                            permut_indices,
                            stacked_object_emb,
                        ) in object_permut_batch:
                            objects_per_atom.append(
                                [state_object_names[index] for index in permut_indices]
                            )
                            flattened_permuted_emb.append(
                                stacked_object_emb.view(1, -1)
                            )
                batched_permuted_emb = torch.cat(flattened_permuted_emb, dim=0)
                batched_repeated_state_aggr = torch.cat(repeated_state_aggr_emb, dim=0)
                assert (
                    batched_permuted_emb.shape[0]
                    == batched_repeated_state_aggr.shape[0]
                ), (
                    f"Batch size mismatch between permuted embeddings and repeated state aggregation. "
                    f"Given: {batched_permuted_emb.shape=}, {batched_repeated_state_aggr.shape=}"
                )
                batched_permuted_emb = torch.cat(
                    (batched_repeated_state_aggr, batched_permuted_emb), dim=1
                )

                for predicate in self.arity_to_predicates[arity]:
                    predicate_out[predicate], output_info[predicate] = (
                        self._forward_predicate(
                            predicate,
                            batched_permuted_emb,
                            state_association,
                            objects_per_atom,
                            with_info=with_info,
                        )
                    )
        if self.device.type == "cuda":
            # Synchronize the CUDA stream to ensure all operations are completed before returning
            torch.cuda.synchronize(self.device)
        if with_info:
            return predicate_out, output_info
        else:
            return predicate_out

    def _forward_predicate(
        self,
        predicate,
        batched_permutations: Tensor,
        state_association: Iterable[int] | None = None,
        atom_objects: Iterable[Sequence[str]] | None = None,
        with_info: bool = False,
    ):
        mlp = self.valuator_by_predicate[predicate]
        with torch.cuda.stream(self.stream_for(predicate)):
            predicate_out = mlp(batched_permutations)
        if with_info:
            output_info = [
                OutputInfo(
                    state_index,
                    atom_str_template.render(predicate=predicate, objects=objects),
                )
                for state_index, objects in zip(state_association, atom_objects)
            ]
            return predicate_out, output_info
        return predicate_out, None
