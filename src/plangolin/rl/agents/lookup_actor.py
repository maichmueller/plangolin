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
        probs_list: list[torch.Tensor],
        problem: XProblem,
        env_keys: PlanningEnvironment.AcceptedKeys,
        idx_of_state: TensorDictModule | XStateSpace | str,
    ):
        # keep custom fields
        match probs_list:
            case torch.nested.Tensor():
                pass
            case list():
                probs_list = torch.tensor(probs_list, dtype=torch.float32)
            case _:
                raise TypeError(
                    f"Expected probs_list to be a list or a 1D tensor, got {type(probs_list)}"
                )
        self.probs_list = probs_list
        self.problem = problem
        self.env_keys = env_keys
        # index -> probs (deterministic blocks)
        if isinstance(idx_of_state, XStateSpace):
            # if we have a space, we can rely on the indices of states to be persistent
            state_index_mod = lambda states: [s.index for s in tolist(states)]
            state_index_mod = TensorDictModule(
                module=state_index_mod,
                in_keys=[env_keys.state],
                out_keys=["state_index"],
            )
        elif isinstance(idx_of_state, str):
            # if we have a string, we assume it's the key in the tensordict
            state_index_mod = TensorDictModule(
                module=torch.nn.Identity(),  # no-op, just pass through
                in_keys=[idx_of_state],
                out_keys=["state_index"],
            )
        elif isinstance(idx_of_state, TensorDictModule):
            # if it's a TensorDictModule, we can use it directly
            assert idx_of_state.out_keys == [
                "state_index"
            ], "idx_of_state must output 'state_index'"
            state_index_mod = idx_of_state
        else:
            raise TypeError(
                f"Expected idx_of_state to be a callable, XStateSpace or str, got {type(idx_of_state)}"
            )
        probs_retriever = TensorDictModule(
            module=lambda indices: [self.probs_list[i] for i in indices],
            in_keys=["state_index"],
            out_keys=["probs"],
        )
        # stochastic actor (probabilistic block)
        policy_head = ProbabilisticTensorDictModule(
            in_keys=["probs"],  # params to build the dist
            out_keys=[
                PlanningEnvironment.AcceptedKeys.action
            ],  # where sampled action goes
            distribution_class=Categorical,
            return_log_prob=True,  # writes "log_prob" into the tensordict
        )
        super().__init__([state_index_mod, probs_retriever, policy_head])
