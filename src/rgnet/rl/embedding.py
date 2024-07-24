from typing import List, Optional

import pymimir as mi
import torch
from tensordict import NestedKey, TensorDictBase
from torch_geometric.data import Batch
from torch_geometric.nn import Aggregation
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import Transform, TransformedEnv

from rgnet.encoding import HeteroGraphEncoder
from rgnet.models import HeteroGNN
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, non_tensor_to_list


class EmbeddingModule(torch.nn.Module):

    def __init__(
        self,
        encoder: HeteroGraphEncoder,
        hidden_size: int,
        num_layer: int,
        aggr: str | Aggregation | None = None,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.device = device
        self.gnn = HeteroGNN(
            hidden_size=hidden_size,
            num_layer=num_layer,
            aggr=aggr,
            obj_type_id=encoder.obj_type_id,
            arity_dict=encoder.arity_dict,
        )
        self.encoder: HeteroGraphEncoder = encoder

    def forward(self, states: List[mi.State] | NonTensorWrapper) -> torch.Tensor:
        states = non_tensor_to_list(states)
        assert isinstance(states, List)

        as_batch = Batch.from_data_list(
            [self.encoder.to_pyg_data(self.encoder.encode(state)) for state in states]
        )
        as_batch = as_batch.to(self.device)
        return self.gnn(as_batch.x_dict, as_batch.edge_index_dict, as_batch.batch_dict)


class EmbeddingTransform(Transform):

    def __init__(
        self,
        current_embedding_key: NestedKey,
        env: ExpandedStateSpaceEnv,
        embedding_module: EmbeddingModule,
        **kwargs,
    ):
        """
        Calculate the embeddings for the current state provided by the env.
        TransformedEnv will call _step and _reset and not forward.
        NOTE: It is not quite clear what forward is supposed to do.
        :param state_key: Under which key the environment stores the current state.
        :param current_embedding_key: Under which key the embeddings will be stored.
        :param env: The expanded state space base environment.
        :param embedding_module: Module to calculate the embeddings.
        :param kwargs: Additional keywords for super class.
        """
        self.state_key: NestedKey = env.keys.state
        self.current_embedding_key = current_embedding_key
        super().__init__(
            in_keys=[self.state_key],
            out_keys=[self.current_embedding_key],
            **kwargs,
        )
        self.env = env
        self.embedding_module = embedding_module

    def _apply_transform(self, states: NonTensorWrapper) -> TensorDictBase:
        """This function will be called by _call for every in-key."""
        return self.embedding_module(states)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """
        We want to produce embeddings fore initial states too. It is not clear to me
        why the parent method does nothing.
        NOTE during a partial reset (inside maybe_reset) this function is called before
        base_env completed _reset_proc_data and hence tensordict_reset will contain
        only states that were reset, of which most will be thrown out again.
        Ideally we could check if "_reset" is present in tensordict and only create embeddings
        for the states that are actually reset.
        """
        return self._call(tensordict_reset)

    def transform_observation_spec(
        self, observation_spec: CompositeSpec
    ) -> CompositeSpec:
        """Add current_embeddings to the observation spec.
        This is important for _StepMDP validate."""
        new_observation_spec = observation_spec.clone()
        embedding_shape: List[int] = list(observation_spec.shape)
        embedding_shape.append(self.embedding_module.hidden_size)
        new_observation_spec[self.current_embedding_key] = (
            UnboundedContinuousTensorSpec(shape=torch.Size(embedding_shape))
        )
        return new_observation_spec


class NonTensorTransformedEnv(TransformedEnv):

    def rand_action(self, tensordict: Optional[TensorDictBase] = None):
        """TransformedEnv does not delegate calls to the base_env and hence overridden
        functions for the base env, like rand_action, will not be called.
        """
        return self.base_env.rand_action(tensordict)
