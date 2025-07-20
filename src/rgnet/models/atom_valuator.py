import itertools
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import torch
import torch_geometric.nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver

from rgnet.utils.batching import batched_permutations
from rgnet.utils.misc import KeyAwareDefaultDict, num_nodes_per_entry, tolist
from xmimir import XCategory, XDomain, XPredicate
from xmimir.wrappers import XAtom, atom_str_template

from .attention_aggr import AttentionPooling
from .mixins import DeviceAwareMixin
from .mlp import ArityMLPFactory
from .patched_module_dict import PatchedModuleDict


class OutputInfo(NamedTuple):
    state_index: int
    atom: str


class AtomValuator(DeviceAwareMixin, torch.nn.Module):
    def __init__(
        self,
        predicates: Iterable[XPredicate] | XDomain,
        predicate_module_factory: Callable[[str, int], torch.nn.Module] | None = None,
        pooling: str = "sum",
        activation: str = "mish",
        feature_size: int = 64,
        assert_output: bool = False,
        do_sync: bool = True,
    ):
        super().__init__()
        if isinstance(predicates, XDomain):
            predicates = predicates.predicates(XCategory.fluent, XCategory.derived)
        # make them hollow to not have to deal with pickling problems
        predicates = [
            XPredicate.make_hollow(p.name, p.arity, p.category)
            for p in predicates
            if not p.is_hollow
        ]
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
        self.max_arity = max(self.arity_dict.values())
        if predicate_module_factory is None:
            predicate_module_factory = ArityMLPFactory(
                feature_size, added_arity=1, layers=3, activation=activation
            )
        self.valuator_by_predicate = PatchedModuleDict(
            {
                # One Module per predicate
                # For a predicate p(o_1,...,o_k) the corresponding Module gets k object
                # embeddings as input and generates k outputs, one for each object.
                pred: torch.nn.Sequential(
                    predicate_module_factory(pred, arity),
                    activation_resolver(activation),
                    torch.nn.Linear(feature_size * (arity + 1), 1),
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
            case "attention":
                self.pooling = AttentionPooling(
                    feature_size=feature_size, num_heads=10, split_features=True
                )
            case _:
                raise ValueError(
                    f"Unknown state pooling function: {pooling}. "
                    f"Choose from [sum, add, max, mean, attention]."
                )
        self.feature_size = feature_size
        self.assert_output = assert_output
        self.streams: dict[str, torch.cuda.Stream] = dict()
        self._do_sync = do_sync

    def stream_for(self, predicate: str) -> torch.cuda.Stream | None:
        if self.device.type == "cpu":
            return None
        if not self.streams and self.device.type == "cuda":
            self.streams = {
                pred: torch.cuda.Stream(self.device) for pred in self.predicates_dict
            }
        return self.streams[predicate]

    @property
    def arities(self) -> Iterable[int]:
        """Returns the set of arities present of the predicates."""
        return self.arity_to_predicates.keys()

    @property
    def predicates(self) -> Iterable[str]:
        """Returns the set of predicates present in the model."""
        return self.predicates_dict.keys()

    def forward(
        self,
        embeddings: Tensor,
        batch_info: Tensor,
        object_names: Sequence[Sequence[str]] | Sequence[str],
        atoms: Sequence[XAtom] | None = None,
        provide_output_metadata: bool | None = None,
        info_dict: dict[str, Any] | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, list[OutputInfo]]]:
        """
        Batch is a PyG Batch object containing the graphs of batch_size many states.

        We compute the embeddings of the objects in the states and split up the tensor into state-respective
        object-embedding pieces, i.e. each state has its own object-embedding 2D-tensor where dim-0 is the object
        dimension and dim-1 the feature dimension.
        We then compute all possible permutations (with replacement) of object-embeddings of size `arity(predicate)`
        in each state and batch them together again.

        The predicate-specific MLP then performs a readout of the atom values.

        :param embeddings: The object embeddings for the batch of states. We assume that the embeddings are already
            computed and passed as input.
            The shape of the embeddings should be:
                (batch_size * avg_object_count_per_state, feature_size).
        :param batch_info: A tensor of len(embeddings) indicating for each entry in the batch which state it belongs to.
        :param object_names: A sequence of len == max(batch_info) containing either...
            - a sequence of object names for each state (heterogeneous case - multiple states from multiple problems),
            - a single list of object names (homogeneous case - multiple states from same problem).
            Implicit assumption is that each object name list corresponds to the same object order in the embeddings,
            e.g. if k is the start index of a state's 6 object embeddings, then the i-th object in the object_names
            corresponds to the (k+i)-th row in the embeddings for 0 <= i < 6.
        :param atoms: Optional sequence of atoms to evaluate. When evaluating specific atoms, the input embeddings
            must be homogeneous.
            If not given, all possible permutations of objects for each predicate will be evaluated.
        :param provide_output_metadata: If True, returns additional information about the output for verification.
        :param info_dict: Additional information dictionary that may contain the batch size or other batch related info.
        """
        if provide_output_metadata is None:
            provide_output_metadata = self.assert_output
        num_objects = num_nodes_per_entry(batch_info)
        nr_states = len(num_objects)
        object_emb_per_state: tuple[Tensor, ...] = torch.split(
            embeddings, tolist(num_objects)
        )
        predicate_out = dict()
        output_info: dict[str, list[OutputInfo]] = dict()
        pooled_emb_batch = self.pooling(
            x=embeddings,
            batch=batch_info,
            size=(info_dict or {}).get("batch_size", None),
        )
        if (
            isinstance(object_names, Sequence)
            and object_names
            and isinstance(object_names[0], str)
        ):
            object_names = [object_names] * nr_states

        if atoms is None:
            predicate_filter = lambda p: True
        else:
            given_predicates = set(atom.predicate.name for atom in atoms)
            assert all(
                predicate in self.predicates_dict for predicate in given_predicates
            ), (
                f"Some atoms in the input are not part of the known predicates: "
                f"{given_predicates - self.predicates_dict.keys()=}"
            )
            predicate_filter = lambda p: p in given_predicates

        atom_permutations = KeyAwareDefaultDict(
            lambda arity: self._all_embedding_permutations(
                object_emb_per_state,
                object_names,
                arity,
                pooled_emb_batch,
            )
        )  # cache for already computed permutations, may be unused

        for predicate, arity in self.arity_dict.items():
            if not predicate_filter(predicate):
                continue

            if arity == 0:
                # there are no objects in the predicate,
                # so we only compute the atom's value from the state aggregation alone
                predicate_out[predicate], output_info[predicate] = (
                    self._forward_predicate(
                        predicate,
                        pooled_emb_batch,
                        range(len(object_emb_per_state)),
                        itertools.repeat([]),
                        provide_output_metadata=provide_output_metadata,
                    )
                )
                continue

            if atoms is None:
                # For all predicates, we compute all possible permutations of objects (this is only arity-specific,
                # so we can reuse the permutations for each predicate of the same arity).
                batched_permuted_emb, state_association, objects_per_atom = (
                    atom_permutations[arity]
                )
            else:
                # For given atoms, we only consider the objects that are part of the atom
                # and permute them accordingly (this is predicate- & arity-specific).
                batched_permuted_emb, state_association, objects_per_atom = (
                    self._only_atom_embeddings(
                        predicate,
                        atoms,
                        object_emb_per_state,
                        object_names,
                        pooled_emb_batch,
                    )
                )

            predicate_out[predicate], output_info[predicate] = self._forward_predicate(
                predicate,
                batched_permuted_emb,
                state_association,
                objects_per_atom,
                provide_output_metadata=provide_output_metadata,
            )

        if self._do_sync and self.device.type == "cuda":
            # Synchronize the CUDA stream to ensure all operations are completed before returning
            for stream in self.streams.values():
                stream.synchronize()
        if provide_output_metadata:
            return predicate_out, output_info
        else:
            return predicate_out

    def _forward_predicate(
        self,
        predicate: str,
        batched_permuted_embeddings: Tensor,
        state_association: Iterable[int] | None = None,
        atom_objects: Iterable[Sequence[str]] | None = None,
        provide_output_metadata: bool = False,
    ):
        mlp = self.valuator_by_predicate[predicate]
        with torch.cuda.stream(self.stream_for(predicate)):
            predicate_out = mlp(batched_permuted_embeddings)
        if provide_output_metadata:
            output_info = [
                OutputInfo(
                    state_index,
                    atom_str_template.render(predicate=predicate, objects=objects),
                )
                for state_index, objects in zip(state_association, atom_objects)
            ]
            return predicate_out, output_info
        return predicate_out, None

    def _all_embedding_permutations(
        self,
        object_emb_per_state: Sequence[Tensor],
        object_names: Sequence[Sequence[str]] | Sequence[str],
        arity,
        aggr_state_embeddings: Tensor,
    ) -> tuple[Tensor, list[int], list[list[str]]]:
        flattened_permuted_emb = []
        objects_per_atom = []
        state_association = []
        repeated_state_aggr_emb = []
        for state_index, (
            state_object_names,
            state_object_embs,
            state_emb,
        ) in enumerate(zip(object_names, object_emb_per_state, aggr_state_embeddings)):
            for permutation_batch in batched_permutations(
                state_object_embs,
                arity,
                batch_size=1,
                with_replacement=True,  # allow repeated objects in atoms as args, e.g. (on a a), (= b b), etc.
            ):
                nr_perms = len(permutation_batch)
                state_association.extend([state_index] * nr_perms)
                repeated_state_aggr_emb.append(
                    state_emb.view(1, -1).expand(nr_perms, -1)
                )
                for permut_indices, stacked_object_emb in permutation_batch:
                    objects_per_atom.append(
                        [state_object_names[index] for index in permut_indices]
                    )
                    flattened_permuted_emb.append(stacked_object_emb.view(1, -1))
        batched_permuted_emb = torch.cat(flattened_permuted_emb, dim=0)
        batched_repeated_state_aggr = torch.cat(repeated_state_aggr_emb, dim=0)
        assert batched_permuted_emb.shape[0] == batched_repeated_state_aggr.shape[0], (
            f"Batch size mismatch between permuted embeddings and repeated state aggregation. "
            f"Given: {batched_permuted_emb.shape=}, {batched_repeated_state_aggr.shape=}"
        )
        batched_permuted_emb = torch.cat(
            (batched_repeated_state_aggr, batched_permuted_emb), dim=1
        )
        return batched_permuted_emb, state_association, objects_per_atom

    def _only_atom_embeddings(
        self,
        predicate: str,
        atoms: Sequence[XAtom],
        object_emb_per_state: Sequence[Tensor],
        object_names: Sequence[Sequence[str]] | Sequence[str],
        aggr_state_embeddings: Tensor,
    ) -> tuple[Tensor, list[int], list[list[str]]]:
        """
        Returns the batched embeddings for the given atoms.
        """
        arity = self.arity_dict[predicate]
        atom_indices, atom_objects = zip(
            *(
                (i, atom.objects)
                for i, atom in enumerate(atoms)
                if atom.predicate.name == predicate
            )
        )
        nr_relevant_atoms = len(atom_indices)
        state_association = sum(
            (
                [state_index] * nr_relevant_atoms
                for state_index in range(len(object_emb_per_state))
            ),
            [],
        )
        flattened_permuted_emb = []
        batched_repeated_state_aggr = aggr_state_embeddings.view(
            -1, self.feature_size
        ).repeat(nr_relevant_atoms, 1)
        for state_object_names, state_object_embs in zip(
            object_names, object_emb_per_state
        ):
            for idx in atom_indices:
                atom = atoms[idx]
                assert (
                    len(atom_objects[idx]) == arity
                ), f"Atom {atom} has wrong arity {len(atom.objects)} != {arity}."
                permut_indices = [
                    state_object_names.index(obj.get_name()) for obj in atom.objects
                ]
                # [1, nr_objects * feature_size]
                stacked_object_emb = state_object_embs[permut_indices].view(1, -1)
                flattened_permuted_emb.append(stacked_object_emb)
        batched_permuted_emb = torch.cat(flattened_permuted_emb, dim=0)
        assert batched_permuted_emb.shape[0] == batched_repeated_state_aggr.shape[0], (
            f"Batch size mismatch between permuted embeddings and repeated state aggregation. "
            f"Given: {batched_permuted_emb.shape=}, {batched_repeated_state_aggr.shape=}"
        )
        batched_permuted_emb = torch.cat(
            (batched_repeated_state_aggr, batched_permuted_emb), dim=1
        )
        return batched_permuted_emb, state_association, atom_objects

    def __getstate__(self):
        # Custom getstate to handle the streams
        state = self.__dict__.copy()
        state["_streams"] = None
        return state

    def __setstate__(self, state):
        # Custom setstate to reinitialize the streams
        self.__dict__.update(state)
        if self.device.type == "cuda":
            self.streams = {
                pred: torch.cuda.Stream(self.device) for pred in self.predicates_dict
            }
        else:
            self.streams = {}
