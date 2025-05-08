from typing import Iterable, NamedTuple

import torch
from jedi.inference.gradual.typing import Callable
from torch import Tensor

from rgnet.utils.batching import batched_permutations
from rgnet.utils.misc import tolist
from xmimir import XDomain, XPredicate
from xmimir.wrappers import atom_str_template

from .hetero_gnn import simple_mlp
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
        activation: str = "mish",
        feature_size: int = 128,
        assert_output: bool = False,
    ):
        super().__init__()
        if isinstance(predicates, XDomain):
            predicates = predicates.predicates()
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
        self.atom_valuators = {
            # One Module per predicate
            # For a predicate p(o_1,...,o_k) the corresponding Module gets k object
            # embeddings as input and generates k outputs, one for each object.
            pred: (
                ResidualModule(
                    simple_mlp(
                        *([feature_size * arity] * 3),
                        activation=activation,
                    )
                )
                if predicate_module_factory is None
                else predicate_module_factory(
                    self.predicates_dict[pred], feature_size=feature_size
                )
            )
            for pred, arity in self.arity_dict.items()
            if arity > 0
        }
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
        num_objects: Tensor | list[int],
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
        :param num_objects: A tensor containing the number of objects in each state.
        :param object_names: A list containing the list of object names in each state.
        :param with_info: If True, returns additional information about the output for verification.
        """
        if with_info is None:
            with_info = self.assert_output
        embeddings_per_state: tuple[Tensor, ...] = torch.split(
            embeddings, tolist(num_objects)
        )
        predicate_out = dict()
        output_info: dict[str, list[OutputInfo]] = dict()
        for arity in self.arity_to_predicates.keys():
            flattened_permutations = []
            atom_objects_of_arity = []
            state_association = []
            for state_index, (state_object_names, state_object_embeddings) in enumerate(
                zip(object_names, embeddings_per_state)
            ):
                stacked_embeds = tuple(
                    emb.squeeze().view(1, -1) for emb in state_object_embeddings
                )
                stacked_embeds = torch.stack(stacked_embeds, dim=0)
                for object_permutation_batch in batched_permutations(
                    stacked_embeds,
                    arity,
                    batch_size=1,
                ):
                    state_association.extend(
                        [state_index] * len(object_permutation_batch)
                    )
                    for (
                        permutation_indices,
                        stacked_object_embeddings,
                        _,
                    ) in object_permutation_batch:
                        atom_objects_of_arity.append(
                            tuple(
                                state_object_names[index]
                                for index in permutation_indices
                            )
                        )
                        flattened_permutations.append(stacked_object_embeddings)
            batched_permuted_embeds = torch.cat(flattened_permutations, dim=0).to(
                self.device
            )
            for predicate in self.arity_to_predicates[arity]:
                mlp = self.atom_valuators[predicate]
                with torch.cuda.stream(self.stream_for(predicate)):
                    predicate_out[predicate] = mlp(batched_permuted_embeds)
                if with_info:
                    output_info[predicate] = [
                        OutputInfo(
                            state_index,
                            atom_str_template.render(
                                predicate=predicate, objects=objects
                            ),
                        )
                        for state_index, objects in zip(
                            state_association, atom_objects_of_arity
                        )
                    ]
        torch.cuda.synchronize(self.device)
        if with_info:
            return predicate_out, output_info
        else:
            return predicate_out
