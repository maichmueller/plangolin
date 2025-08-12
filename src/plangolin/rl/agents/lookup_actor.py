from typing import Sequence

import torch
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModule,
    TensorDictSequential,
)
from torch.distributions import Categorical

from plangolin.rl.envs import PlanningEnvironment
from plangolin.utils.misc import tolist
from xmimir import XProblem, XStateSpace


class LookupPolicyActor(TensorDictSequential):
    """TorchRL-compatible stochastic policy sampling from a pre-computed per-state probs list."""

    def __init__(
        self,
        probs_list: Sequence[torch.Tensor],
        problem: XProblem,
        env_keys: PlanningEnvironment.AcceptedKeys,
        idx_of_state: TensorDictModule | XStateSpace | str,
    ):
        # Normalize per-state probability vectors into a dense logits table [N, A_max]
        if isinstance(probs_list, torch.Tensor) and getattr(
            probs_list, "is_nested", False
        ):
            nested = probs_list
        elif isinstance(probs_list, Sequence):
            assert len(probs_list) > 0 and isinstance(probs_list[0], torch.Tensor), (
                "Expected probs_list to be a sequence of tensors, got a sequence of "
                f"{type(probs_list[0])}s."
            )
            nested = torch.nested.nested_tensor(list(probs_list), dtype=torch.float32)
        else:
            raise TypeError(
                f"Expected probs_list to be a nested tensor or a sequence of 1D tensors, got {type(probs_list)}"
            )
        padded = nested.to_padded_tensor(0.0)
        # Compute logits; padded zeros become -inf to mask invalid actions
        logits_table = padded.clamp_min(1e-12).log()
        logits_table = torch.where(
            padded == 0.0, torch.full_like(logits_table, float("-inf")), logits_table
        )
        self.logits_table = logits_table
        self.problem = problem
        self.env_keys = env_keys
        # index -> probs (deterministic blocks)
        if isinstance(idx_of_state, XStateSpace):
            # if we have a space, we can rely on the indices of states to be persistent, so we simply
            # use the indices of the states in the tensordict itself.
            def _to_index(states):
                return torch.as_tensor(
                    [s.index for s in tolist(states)], dtype=torch.long
                )

            state_index_mod = TensorDictModule(
                module=_to_index,
                in_keys=[env_keys.state],
                out_keys=["state_index"],
            )
        elif isinstance(idx_of_state, str):
            # if we have a string, we assume it's the key in the tensordict
            def _as_long(x):
                return x.to(torch.long)

            state_index_mod = TensorDictModule(
                module=_as_long,  # cast to indices
                in_keys=[idx_of_state],
                out_keys=["state_index"],
            )
        elif isinstance(idx_of_state, TensorDictModule):
            # if it's a TensorDictModule, we can use it directly
            assert "state_index" in [
                str(k) for k in idx_of_state.out_keys
            ], "idx_of_state must output 'state_index'"
            state_index_mod = idx_of_state
        else:
            raise TypeError(
                f"Expected idx_of_state to be a callable, XStateSpace or str, got {type(idx_of_state)}"
            )

        def _gather_logits(indices):
            if isinstance(indices, torch.Tensor):
                idx = indices.to(torch.long).reshape(-1)
                rows = self.logits_table.index_select(0, idx)
                return rows.to(indices.device)
            idx = torch.as_tensor([int(i) for i in indices], dtype=torch.long)
            return self.logits_table.index_select(0, idx)

        logits_retriever = TensorDictModule(
            module=_gather_logits,
            in_keys=["state_index"],
            out_keys=["logits"],
        )
        # stochastic actor (probabilistic block)
        policy_head = ProbabilisticTensorDictModule(
            in_keys=["logits"],  # params to build the dist
            out_keys=[env_keys.action],  # where sampled action goes
            distribution_class=Categorical,
            return_log_prob=True,  # writes "log_prob" into the tensordict
        )
        super().__init__([state_index_mod, logits_retriever, policy_head])
