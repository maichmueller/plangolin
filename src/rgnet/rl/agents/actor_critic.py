import dataclasses
import itertools
from typing import List, Optional, Tuple

import pymimir as mi
import torch
from tensordict import NestedKey, NonTensorStack, TensorDict
from tensordict.nn import ProbabilisticTensorDictModule, TensorDictModule
from torch import Tensor
from torch_geometric.nn.models import MLP
from torchrl.modules.tensordict_module import ValueOperator

from rgnet.rl.embedding import EmbeddingModule
from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


def embed_transition_targets(
    batched_transitions: List[List[mi.Transition]], embedding_module: EmbeddingModule
) -> Tuple[Tensor, ...]:
    """
    Calculate embeddings for the targets for each transition. This will only
    trigger one call to the embedding module by flattening the batch beforehand.
    :param batched_transitions: We expect the transitions to be batched as a list.
    :return: the embedded targets of shape [batch_size x num_successors]
    """
    flattened = list(
        itertools.chain.from_iterable(
            [t.target for t in transitions] for transitions in batched_transitions
        )
    )
    long_tensor: torch.Tensor = embedding_module(flattened)
    start_indices = list(itertools.accumulate(map(len, batched_transitions)))
    # tensor_split will split at every index -> we need to remove last index,
    # which is the length of the tensor
    start_indices = start_indices[:-1]
    return long_tensor.tensor_split(start_indices, dim=0)


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
        embedding_module: EmbeddingModule,
        value_net: torch.nn.Module | None = None,
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

        self._hidden_size = embedding_module.hidden_size

        self._embedding_module = embedding_module

        self.probabilistic_module = ProbabilisticTensorDictModule(
            in_keys=[self._keys._distr_key],
            out_keys=[self._keys._action_idx_key],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
            log_prob_key=self._keys.log_probs,
            n_empirical_estimate=0,
        )

        self.actor_net = torch.nn.Sequential(
            # Input: embeddings of current state and next state, Output: 2 + hidden size
            MLP(
                channel_list=[2 * self._hidden_size] * 2,  # all three layer are same
                norm=None,
                dropout=0.0,
            ),
            # Input: 2 * hidden size, Output: single scalar "logits"
            MLP(
                channel_list=[2 * self._hidden_size, 2 * self._hidden_size, 1],
                norm=None,
                dropout=0.0,
            ),
        )

        # The ValueOperator is the critic of the actor-critic approach.
        # provided with the embeddings of the current state it estimates the value.
        if value_net is None:
            value_net = MLP(
                channel_list=[self._hidden_size, self._hidden_size, 1],
                norm=None,
                dropout=0.0,
            )
        self.value_operator = ValueOperator(
            module=value_net,
            in_keys=[self._keys.current_embedding],
            out_keys=[self._keys.state_value],
        )

    @property
    def keys(self):
        return self._keys

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
            actor_net. The length of the list is the batch_size.
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
        current_embeddings: torch.Tensor,
        successor_embeddings: Tuple[torch.Tensor, ...],
    ) -> List[torch.Tensor]:
        assert isinstance(successor_embeddings, Tuple)

        # First dimension is batch, second number of successors, third hidden_size
        # Number of successors can be different for each state -> list instead of tensor
        # Assert that the batch dimension is the same
        assert current_embeddings.shape[0] == len(successor_embeddings)
        # Assert that they have the same embedding size
        assert current_embeddings.shape[-1] == successor_embeddings[0].shape[-1]
        # 1. Compute pairs of (current_embeddings[i], successor_embedding)
        #    for each successor_embedding in successor_embeddings[i]
        # 2. Run self.actor_net on each pair
        # 3. Compute the softmax over all outputs of 2.
        probabilities_batched: List[torch.Tensor] = []  # batch_size x num_successors
        for i in range(current_embeddings.shape[0]):  # loop over the batch
            num_successor = successor_embeddings[i].shape[0]
            # Use expand instead of repeat as we only need a view and not a copy
            # We expand a new axis, a.k.a the num_successor axis of successors
            # New shape num_successor x hidden_size
            expanded_current = current_embeddings[i].expand(num_successor, -1)
            # Create the pairs of current_embedding, successor_embedding
            pairs = torch.cat(
                (expanded_current, successor_embeddings[i]),
                dim=1,
            )
            # flatten to reduce shape = num_successors x 1 to shape = num_successor
            # Here we convert logits to probabilities by normalizing using softmax
            probs_successors = self.actor_net(pairs).flatten().softmax(dim=0)
            probabilities_batched.append(probs_successors)

        return probabilities_batched

    def forward(
        self,
        state: NonTensorWrapper | List[mi.State],
        transitions: NonTensorWrapper | List[List[mi.Transition]],
        current_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[
        NonTensorStack, torch.Tensor, torch.Tensor, NonTensorStack, NonTensorStack
    ]:

        transitions: List[List[mi.Transition]] = non_tensor_to_list(transitions)
        assert all(
            len(ts) > 0 for ts in transitions
        ), "Found empty transition, environment should reset on dead-end states."
        if current_embedding is None:
            current_embedding = self._embedding_module(state)
        successor_embeddings: Tuple[Tensor, ...] = embed_transition_targets(
            transitions, self._embedding_module
        )
        # len(batched_probs) == batch_size, batched_probs[i].shape == len(transitions[i])
        batched_probs: list[Tensor] = self._actor_probs(
            current_embedding, successor_embeddings
        )

        action_indices, log_probs = self._sample_distribution(batched_probs)

        actions = self._select_action(action_indices, transitions)

        return (
            as_non_tensor_stack(actions),
            current_embedding,
            log_probs,
            as_non_tensor_stack(batched_probs),
            as_non_tensor_stack(successor_embeddings),
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
            out_keys.append(self._keys.successor_embeddings)
        return TensorDictModule(
            module=self,
            in_keys=[state_key, transition_key, self._keys.current_embedding],
            out_keys=out_keys,
        )
