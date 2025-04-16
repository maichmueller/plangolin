from __future__ import annotations

from typing import List, Optional

import torch
from tensordict import NestedKey, TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import Transform, TransformedEnv

from rgnet.rl.embedding.embedding_module import EmbeddingModule
from rgnet.rl.envs import PlanningEnvironment
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper


class EmbeddingTransform(Transform):
    def __init__(
        self,
        current_embedding_key: NestedKey,
        env: PlanningEnvironment,
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
        return self.embedding_module(states).to_tensordict()

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

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        """Add current_embeddings to the observation spec.
        This is important for _StepMDP validate."""
        new_observation_spec = observation_spec.clone()
        embedding_shape: List[int] = list(observation_spec.shape)
        embedding_shape.append(self.embedding_module.hidden_size)
        device = self.embedding_module.device
        new_observation_spec[self.current_embedding_key] = Composite(
            dense_embedding=Unbounded(shape=torch.Size(embedding_shape), device=device),
            padding_mask=Unbounded(shape=torch.Size(embedding_shape), device=device),
            shape=observation_spec.shape,
            device=device,
        )
        return new_observation_spec


class NonTensorTransformedEnv(TransformedEnv):
    def rand_action(self, tensordict: Optional[TensorDictBase] = None):
        """TransformedEnv does not delegate calls to the base_env and hence overridden
        functions for the base env, like rand_action, will not be called.
        """
        return self.base_env.rand_action(tensordict)
