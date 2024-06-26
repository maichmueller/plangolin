import dataclasses
from typing import List, Tuple

import pymimir as mi
import torch
import torch.nn.functional as F
from tensordict import NestedKey, NonTensorStack
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch_geometric.nn.models import MLP
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ValueEstimators

from rgnet.rl.embedding import EmbeddingModule, embed_states_and_transitions
from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


class Agent:

    @dataclasses.dataclass
    class _AcceptedKeys:
        """Maintains the default values for all steps within the agent pipeline.
        Attributes:
            state : The input key under which the batched current state is stored.
            transitions: The input key under which the batched transitions are stored.
            current_embedding: The key for the embeddings of the current states.
            successor_embedding: The key for the embeddings of the transition-targets.
            action_idx: The key for index of the selected action, as sampled by the actor.
            state_value: The key for the state value, estimated by the critic.
            action: The key for the chosen action (transition), should most likely match what the environment expects.

        """

        state: NestedKey = "state"
        transitions: NestedKey = "transitions"
        current_embedding: NestedKey = "current_embedding"
        successor_embedding: NestedKey = "successor_embedding"
        action_idx: NestedKey = "action_idx"
        state_value: NestedKey = "state_value"
        action: NestedKey = "action"

    default_keys = _AcceptedKeys()

    def set_keys(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self._AcceptedKeys.__dict__:
                raise ValueError(f"{key} is not an accepted tensordict key")
            if value is not None:
                setattr(self._keys, key, value)
            else:
                setattr(self._keys, key, self.default_keys.key)

    def __init__(
        self,
        embedding_module: EmbeddingModule,
        value_estimator_type: ValueEstimators = ValueEstimators.TD0,
        value_estimator_kwargs=None,
        loss_kwargs=None,
    ):
        """
        The Agent class creates all components necessary for an actor-critic policy
        including a suitable loss function.

        Attributes:
            embedding_module (EmbeddingModule): a module which can generate embeddings
                of states (typically done with the use of a GNN).
            value_estimator_type (ValueEstimators, optional): which value estimator to use for the A2C loss.
                Defaults to ValueEstimators.TD0.
            value_estimator_kwargs: The arguments for the ValueEstimator as passed to the A2C loss.
                Defaults to {'gamma': 0.9}
        """
        if value_estimator_kwargs is None:
            value_estimator_kwargs = {"gamma": 0.9}

        self._keys = self._AcceptedKeys()

        self._hidden_size = embedding_module.hidden_size

        self.embedding_td_module = embed_states_and_transitions(
            embedding_module,
            states_key=self._keys.state,
            transitions_key=self._keys.transitions,
            out_keys=[self._keys.current_embedding, self._keys.successor_embedding],
        )

        self.actor_net = torch.nn.Sequential(
            # Input: embeddings of current state and next state, Output: 2 + hidden size
            MLP(
                channel_list=[2 * self._hidden_size] * 3,  # all three layer are same
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

        # The ProbabilisticActor is the actor of the actor-critic approach.
        # It takes the embeddings as input, calculates logits for each pair
        # (current_embedding,successor_embedding) over all successors and
        # samples an index based on the logits.
        actor_module = TensorDictModule(
            self._policy_function,
            in_keys=[self._keys.current_embedding, self._keys.successor_embedding],
            out_keys=["logits"],  # has to be fixed for the distribution
        )
        self.prob_actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],  # keyword argument for the distribution
            out_keys=[self._keys.action_idx],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )

        # The ValueOperator is the critic of the actor-critic approach.
        # provided with the embeddings of the current state it estimates the value.
        self.value_operator = ValueOperator(
            module=MLP(
                channel_list=[self._hidden_size, self._hidden_size, 1],
                norm=None,
                dropout=0.0,
            ),
            in_keys=[self._keys.current_embedding],
            out_keys=[self._keys.state_value],
        )

        # After sampling an index with torch.distributions.Categorical we need to
        # actually select the element of that index from the input transitions
        self.action_selector = TensorDictModule(
            Agent._select_action,
            in_keys=[self._keys.action_idx, self._keys.transitions],
            out_keys=[self._keys.action],
        )

        # The policy is a pipeline of all the components necessary to produce an
        # action given the state and transitions. It can for example be used to
        # compute rollouts like environment.rollout(10, agent.policy)
        self.policy = TensorDictSequential(
            self.embedding_td_module, self.prob_actor, self.action_selector
        )
        loss_kwargs = loss_kwargs or {}
        self.loss = A2CLoss(
            actor_network=self.prob_actor,
            critic_network=self.value_operator,
            functional=False,
            **loss_kwargs,
        )
        self.loss.set_keys(action=self._keys.action_idx)
        self.loss.make_value_estimator(value_estimator_type, **value_estimator_kwargs)

    def _policy_function(
        self,
        current_embeddings: torch.Tensor,
        successor_embeddings: List[torch.Tensor] | NonTensorWrapper,
    ) -> torch.Tensor:  # logits are padded with 0 in order to get homogenous shape
        """
        Calculate the logits for all pairs (current state, successor state).
        :param current_embeddings: batched embeddings for the current state.
            Shape batch_size x hidden_size.
        :param successor_embeddings: batched embeddings for each successor state for
            each current state. Shape batch_size x num_successor x hidden_size.
        :return: A tensor of shape batch_size x max_number_successors, where the second
            dimension is the greatest number of successors over the entire batch.

        """
        successor_embeddings = non_tensor_to_list(successor_embeddings)

        assert isinstance(successor_embeddings, List)

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
        max_number_successors = max(se.shape[0] for se in successor_embeddings)

        logits_batched: List[torch.Tensor] = []  # batch_size x num_successors
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
            logits_successors = self.actor_net(pairs).flatten().softmax(dim=0)
            logits_batched.append(logits_successors)

        assert all(t.dim() == 1 for t in logits_batched)

        # We need to return a tensor as torch.distributions.Categorical expects a
        # tensor and torchrl just passes the output of this function to Categorical.
        # Therefore, we create a homogenous shape by adding trailing zeros, which will
        # never be sampled by the distribution.
        paddings = [max_number_successors - t.shape[0] for t in logits_batched]
        # shape = batch_size x max_number_successors
        batch_tensor = torch.stack(
            [
                F.pad(t, (0, pad), value=0.0)  # add trailing 0s
                for (pad, t) in zip(paddings, logits_batched)
            ]
        )

        return batch_tensor

    @staticmethod
    def _select_action(
        action_idx: torch.Tensor, transitions: List[List[mi.Transition]]
    ) -> Tuple[NonTensorStack]:  # List[mi.Transition]

        transitions = non_tensor_to_list(transitions)

        assert action_idx.dim() == 1
        assert len(transitions) == len(action_idx)

        # Select the transition with the sampled index
        # NonTensorStack.select(...) only works with keys -> TensorDictBase.select
        # We have to return a single item tuple because TensorDictModule.forward would
        # otherwise try to retrieve the output-key from the NonTensorStack (Line 1204).
        # Open issue https://github.com/pytorch/tensordict/issues/821
        return (
            as_non_tensor_stack(
                [t[idx.item()] for t, idx in zip(transitions, action_idx)]
            ),
        )
