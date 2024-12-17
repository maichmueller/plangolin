from __future__ import annotations

import dataclasses
import itertools
from typing import List, Optional, Tuple

import pymimir as mi
import torch
import torch_geometric as pyg
from tensordict import NestedKey, NonTensorStack, TensorDict
from tensordict.nn import ProbabilisticTensorDictModule, TensorDictModule
from torch import Tensor
from torchrl.modules.tensordict_module import ValueOperator

from rgnet.models.hetero_gnn import simple_mlp
from rgnet.rl.embedding import EmbeddingModule
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, as_non_tensor_stack, tolist
from rgnet.utils.object_embeddings import (
    ObjectEmbedding,
    ObjectPoolingModule,
    mask_to_batch_indices,
)


def embed_transition_targets(
    batched_transitions: List[List[mi.Transition]], embedding_module: EmbeddingModule
) -> ObjectEmbedding:
    """
    Calculate embeddings for the targets for each transition. This will only
    trigger one call to the embedding module by flattening the batch beforehand.
    :param batched_transitions: We expect the transitions to be batched as a list.
    :param embedding_module: The module to generate the embeddings.
    :return: The dense object embeddings. NOTE they are not separated by predecessor state anymore.
    """
    flattened = list(
        itertools.chain.from_iterable(
            [t.target for t in transitions] for transitions in batched_transitions
        )
    )
    return embedding_module(flattened)


class ActorCritic(torch.nn.Module):
    @dataclasses.dataclass(frozen=True)
    class AcceptedKeys:
        # Keys used for the output
        log_probs: NestedKey = "log_probs"  # the log_prob of the taken action
        probs: NestedKey = "probs"  # the probability for each possible action
        state_value: NestedKey = "state_value"  # the output of the value-operator
        current_embedding: NestedKey = "current_embedding"  # the embeddings of states
        successor_embeddings: NestedKey = (
            "successor_embeddings"  # the embeddings of the successors
        )
        # Keys used internally
        _distr_key: NestedKey = "probs"  # the input for the Categorical distribution
        _action_idx_key: NestedKey = "action_idx"  # the output for the distribution

    default_keys = AcceptedKeys()

    def __init__(
        self,
        hidden_size: int,
        embedding_module: Optional[EmbeddingModule] = None,
        value_net: torch.nn.Module | None = None,
        activation: str = "mish",
        keys: AcceptedKeys = default_keys,
    ):
        """
        The Agent class creates all components necessary for an actor-critic policy.
        Due to high amount of bugs in the beta and the untypical use-case we have
        we do not use ProbabilisticActor with only TensorDictModules.

        Attributes:
            embedding_module (EmbeddingModule): a module which can generate embeddings
                of states (typically done with the use of a GNN).
        """
        super().__init__()

        self._keys: ActorCritic.Acceptedkeys = keys

        self._hidden_size = hidden_size

        self._embedding_module = embedding_module

        self.probabilistic_module = ProbabilisticTensorDictModule(
            in_keys=[self._keys._distr_key],
            out_keys=[self._keys._action_idx_key],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
            log_prob_key=self._keys.log_probs,
            n_empirical_estimate=0,
        )

        # Input: object embedding of current state and next state, Output: 2 + hidden size
        # -> Two Linear layer with Mish activation
        self.actor_objects_net = simple_mlp(
            in_size=2 * self._hidden_size,
            hidden_size=2 * self._hidden_size,
            out_size=2 * self._hidden_size,
            activation=activation,
        )

        # Input: 2 * hidden size, Output: single scalar "logits"
        # -> Three Linear layer with one Mish activation
        self.actor_net_probs = simple_mlp(
            in_size=2 * self._hidden_size,
            hidden_size=2 * self._hidden_size,
            out_size=1,
            activation=activation,
        )

        # The ValueOperator is the critic of the actor-critic approach.
        # provided with the embeddings of the current state it estimates the value.
        if value_net is None:
            value_net = simple_mlp(
                in_size=hidden_size,
                hidden_size=hidden_size,
                out_size=1,
                activation=activation,
            )
        self.object_pooling = ObjectPoolingModule("add")
        self.value_operator = ValueOperator(
            module=torch.nn.Sequential(self.object_pooling, value_net),
            in_keys=[self._keys.current_embedding],
            out_keys=[self._keys.state_value],
        )

    @property
    def keys(self):
        return self._keys

    @property
    def embedding_module(self):
        return self._embedding_module

    @staticmethod
    def _select_action(
        action_idx: torch.Tensor, transitions: List[List[mi.Transition]]
    ) -> List[mi.Transition]:

        assert len(transitions) == len(action_idx)
        # NOTE with .tolist() we assume that the transitions are on the CPU
        return [t[idx] for t, idx in zip(transitions, action_idx.tolist())]

    def _sample_distribution(
        self, batched_probs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward the normalized probabilities to the ProbabilisticTensorDictModule.
        This would normally be done inside the ProbabilisticActor, however as  we have
        a ragged tensor across the batch and time dimension we use lists of tensors
        and sample each action-index with a separate call to the probabilistic_module.
        :param batched_probs: A list of tensors containing the normalized logits of the
            actor_net_probs. The length of the list is the batch_size.
        :return: The sampled action_index and the log_probs of chosen sample.
            The action_idx will be list of tensors, which are effectively a single int.
        """

        # td[self._keys._action_idx_key] will actually be a one-element tensor
        sample_indices: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        for probs in batched_probs:
            td = TensorDict({self._keys._distr_key: probs})
            self.probabilistic_module(td)
            sample_indices.append(td[self._keys._action_idx_key])
            log_probs.append(td[self._keys.log_probs])

        return torch.stack(sample_indices), torch.stack(log_probs)

    def _actor_probs(
        self,
        current_embeddings: ObjectEmbedding,
        successor_embeddings: ObjectEmbedding,
        num_successors: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Compute the transition probabilities for each state in the batch.
        We start with the dense, masked representations of the embeddings.
        It is assumed that the successor embeddings are sorted by state.
        The probabilities are computed as in "Learning General Policies with Policy Gradient Methods".
        .. math::
         logits(s'\mid s) = \text{actor_net_probs}(\sum_{o \in O} \text{actor_objects_net}(f^s(o), f^{s'}(o) )
         \pi(s'\mid s) = \propto (logits(s'\mid s))


        :param current_embeddings: The embeddings of the current states.
            The shape should be [batch_size, max_num_objects, hidden_size].
        :param successor_embeddings: The embeddings of all successor states, not separated by state!
            The shape should be [num_successors.sum(), max_num_objects, hidden_size].
            The objects of successor states should always equal the objects of the current state.
        :param num_successors: The number of successors for each state in the batch.
            This will be used to split the successor_embeddings tensor.
        :return: A list of tensors containing the probabilities for each transition for each state.
            The tensors will be of shape [num_successors[i],] for each state i.
        """
        assert isinstance(current_embeddings, ObjectEmbedding)
        assert isinstance(successor_embeddings, ObjectEmbedding)
        assert successor_embeddings.dense_embedding.size(0) == num_successors.sum()
        # Same number of objects and hidden size.
        assert (
            current_embeddings.dense_embedding.shape[1:]
            == successor_embeddings.dense_embedding.shape[1:]
        )

        dense, padding_mask = (
            current_embeddings.dense_embedding,
            current_embeddings.padding_mask,
        )
        dense_successor, is_real_successor = (
            successor_embeddings.dense_embedding,
            successor_embeddings.padding_mask,
        )
        # repeat the current embeddings for each successor.
        repeated_dense = dense.repeat_interleave(repeats=num_successors, dim=0)

        pairs_with_fake = torch.cat([repeated_dense, dense_successor], dim=2)
        pairs = pairs_with_fake[is_real_successor]
        # [N, 2*hidden]
        object_diffs: torch.Tensor = self.actor_objects_net(pairs)

        successor_batch = mask_to_batch_indices(is_real_successor)
        # [batch_size * num_successor, 2 * hidden]
        aggregated_embeddings: torch.Tensor = pyg.nn.global_add_pool(
            object_diffs, successor_batch
        )
        successor_tuple: tuple[torch.Tensor, ...] = aggregated_embeddings.tensor_split(
            num_successors.cpu().cumsum(dim=0)[:-1]
        )
        # List=batch_size, tensors of shape [num_successor,]
        probabilities_batched: list[torch.Tensor] = [
            self.actor_net_probs(successor).flatten().softmax(dim=0)
            for successor in successor_tuple
        ]
        return probabilities_batched

    def embedded_forward(
        self,
        current_embedding: ObjectEmbedding,
        successor_embeddings: ObjectEmbedding,
        num_successors: torch.Tensor,
    ) -> Tuple[List[Tensor], Tensor, Tensor]:
        # len(batched_probs) == batch_size, batched_probs[i].shape == len(transitions[i])
        batched_probs: List[Tensor] = self._actor_probs(
            current_embedding, successor_embeddings, num_successors=num_successors
        )

        action_indices, log_probs = self._sample_distribution(batched_probs)
        return batched_probs, action_indices, log_probs

    def forward(
        self,
        state: NonTensorWrapper | List[mi.State],
        transitions: NonTensorWrapper | List[List[mi.Transition]],
        current_embedding: Optional[ObjectEmbedding | TensorDict] = None,
    ) -> Tuple[NonTensorStack, TensorDict, torch.Tensor, NonTensorStack]:

        transitions: List[List[mi.Transition]] = tolist(transitions)
        assert all(
            len(ts) > 0 for ts in transitions
        ), "Found empty transition, environment should reset on dead-end states."
        if current_embedding is None:
            current_embedding: ObjectEmbedding = self._embedding_module(state)
        elif isinstance(current_embedding, TensorDict):
            current_embedding = ObjectEmbedding.from_tensordict(current_embedding)

        successor_embeddings: ObjectEmbedding = embed_transition_targets(
            transitions, self._embedding_module
        )

        # We need num_successors both on the GPU and CPU default to GPU.
        num_successors = torch.tensor(
            list(map(len, transitions)),
            dtype=torch.long,
            device=self.embedding_module.device,
        )

        batched_probs, action_indices, log_probs = self.embedded_forward(
            current_embedding, successor_embeddings, num_successors=num_successors
        )

        actions = self._select_action(action_indices, transitions)

        return (
            as_non_tensor_stack(actions),
            current_embedding.to_tensordict(),
            log_probs,
            as_non_tensor_stack(batched_probs),
        )

    def as_td_module(
        self,
        state_key: NestedKey,
        transition_key: NestedKey,
        action_key: NestedKey,
        add_probs: bool = False,
        out_successor_embeddings: bool = False,
    ):
        out_keys = [
            action_key,
            self._keys.current_embedding,
            self._keys.log_probs,
        ]
        if add_probs:
            out_keys.append(self._keys.probs)
        if out_successor_embeddings:
            raise RuntimeError("Not supported at the moment")
        return TensorDictModule(
            module=self,
            in_keys=[state_key, transition_key, self._keys.current_embedding],
            out_keys=out_keys,
        )
