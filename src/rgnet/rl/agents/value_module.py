from typing import Callable, List, Tuple, Union

import pymimir as mi
import torch.nn
from tensordict.nn import TensorDictModule
from torch import Tensor
from torch_geometric.nn import MLP

from rgnet.rl.agents.actor_critic import embed_transition_targets
from rgnet.rl.embedding import EmbeddingModule
from rgnet.utils.object_embeddings import ObjectEmbedding, ObjectPoolingModule
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, as_non_tensor_stack, tolist


class ValueModule(torch.nn.Module):

    def __init__(
        self,
        embedding: EmbeddingModule,
        value_net: torch.nn.Module | None = None,
        pooling: Union[str, Callable[[Tensor, Tensor], Tensor]] = "add",
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
        self.value_net = torch.nn.Sequential(ObjectPoolingModule(pooling), value_net)

    def forward(
        self, transitions_in: List[List[mi.Transition]] | NonTensorWrapper
    ) -> List[mi.Transition] | NonTensorWrapper:
        transitions: List[List[mi.Transition]] = tolist(transitions_in)
        with torch.no_grad():
            # We don't want gradient for the next values and next embeddings.
            # The value net will be updated by a ValueEstimator like TD0Estimator.
            successor_embeddings: ObjectEmbedding = embed_transition_targets(
                transitions, self._embedding_module
            )
            successor_values_flat: Tensor = self.value_net(successor_embeddings)
            num_successors = torch.tensor([len(ts) for ts in transitions])
            split_indices = num_successors.cumsum(0)[:-1]
            successor_values: Tuple[Tensor, ...] = successor_values_flat.tensor_split(
                split_indices
            )

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
