import dataclasses
from typing import List, Optional

import pymimir as mi
import torch.nn
from tensordict import NestedKey
from tensordict.nn import TensorDictModule
from torch import nn
from torch_geometric.nn import MLP
from torchrl.envs.utils import ExplorationType, exploration_type

from rgnet.rl import EmbeddingModule
from rgnet.rl.agent import embed_transition_targets
from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


class EGreedyAgent(torch.nn.Module):
    @dataclasses.dataclass
    class _AcceptedKeys:
        pass

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        embedding: EmbeddingModule,
        eps_init: float,
        eps_end: float,
        annealing_num_steps: int,
        value_net: nn.Module | None = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._embedding_module = embedding
        self._hidden_size = self._embedding_module.hidden_size
        self._value_net: nn.Module = value_net or MLP(
            channel_list=[self._hidden_size, self._hidden_size, 1],
            norm=None,
            dropout=0.0,
        )
        self.epsilon = eps_init
        self.eps_init = eps_init
        self.eps_end = eps_end
        self.annealing_steps = annealing_num_steps
        self.keys = EGreedyAgent._AcceptedKeys()

    def step_epsilon(self):
        self.epsilon = max(
            self.eps_end,
            (self.epsilon - (self.eps_init - self.eps_end) / self.annealing_steps),
        )

    def forward(
        self,
        transitions: NonTensorWrapper | List[List[mi.Transition]],
    ):
        transitions = non_tensor_to_list(transitions)

        successor_embeddings = embed_transition_targets(
            transitions, self._embedding_module
        )
        with torch.no_grad():
            successor_values: List[torch.Tensor] = [
                self._value_net(e) for e in successor_embeddings
            ]
        indices_of_best: List[torch.Tensor] = [
            torch.argmax(sv, dim=0) for sv in successor_values
        ]
        actions: List[mi.Transition] = [
            ts[indices_of_best[batch_idx]] for (batch_idx, ts) in enumerate(transitions)
        ]
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            random_steps = torch.rand(size=(len(transitions),)) < self.epsilon
            for batch_idx in range(len(transitions)):
                if random_steps[batch_idx].item():
                    rand_transition_idx = torch.randint(
                        0, len(transitions[batch_idx]), (1,)
                    )
                    actions[batch_idx] = transitions[batch_idx][rand_transition_idx]

            self.step_epsilon()

        return as_non_tensor_stack(actions)

    def as_td_module(self, transition_key: NestedKey, action_key: NestedKey):
        return TensorDictModule(
            module=self,
            in_keys=[transition_key],
            out_keys=[action_key],
        )
