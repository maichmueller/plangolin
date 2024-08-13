from typing import List, Tuple

import pymimir as mi
import torch.nn
from tensordict.nn import TensorDictModule
from torch_geometric.nn import MLP

from rgnet.rl.agents.agent import embed_transition_targets
from rgnet.rl.embedding import EmbeddingModule
from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


class ValueModule(torch.nn.Module):

    def __init__(
        self,
        embedding: EmbeddingModule,
        value_net: torch.nn.Module | None = None,
    ):
        super().__init__()
        self._embedding_module = embedding
        if value_net is None:
            value_net = MLP(
                channel_list=[
                    self._embedding_module.hidden_size,
                    self._embedding_module.hidden_size,
                    1,
                ],
                norm=None,
                dropout=0.0,
            )
        self.value_net = value_net

    def forward(
        self, transitions_in: List[List[mi.Transition]] | NonTensorWrapper
    ) -> List[mi.Transition] | NonTensorWrapper:
        transitions: List[List[mi.Transition]] = non_tensor_to_list(transitions_in)
        with torch.no_grad():
            # We don't want gradient for the next values and next embeddings.
            # The value net will be updated by a ValueEstimator like TD0Estimator.
            successor_embeddings: Tuple[torch.Tensor, ...] = embed_transition_targets(
                transitions, self._embedding_module
            )
            successor_values: List[torch.Tensor] = [
                self.value_net(e) for e in successor_embeddings
            ]
            indices_of_best: List[int] = [
                torch.argmax(sv, dim=0).item() for sv in successor_values
            ]
            actions: List[mi.Transition] = [
                ts[idx_of_best]
                for (idx_of_best, ts) in zip(indices_of_best, transitions)
            ]
        return as_non_tensor_stack(actions)

    def as_td_module(self, transitions_key, actions_key):
        return TensorDictModule(
            module=self, in_keys=[transitions_key], out_keys=[actions_key]
        )
