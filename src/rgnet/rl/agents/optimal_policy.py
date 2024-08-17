from typing import Dict, List

import pymimir as mi
from tensordict import NestedKey
from tensordict.nn import TensorDictModule

from rgnet.rl.non_tensor_data_utils import (
    NonTensorWrapper,
    as_non_tensor_stack,
    non_tensor_to_list,
)


def optimal_action(space: mi.StateSpace, state: mi.State) -> mi.Transition:
    return min(
        space.get_forward_transitions(state),
        key=lambda t: space.get_distance_to_goal_state(t.target),
    )


class OptimalPolicy:

    def __init__(self, space: mi.StateSpace):
        self.space = space
        self.best_actions: Dict[mi.State, mi.Transition] = {
            s: optimal_action(space, s) for s in space.get_states()
        }

    def __call__(
        self, batched_states: List[List[mi.State]] | NonTensorWrapper
    ) -> List[mi.Transition] | NonTensorWrapper:
        batched_states = non_tensor_to_list(batched_states)
        return as_non_tensor_stack(
            [self.best_actions[state] for state in batched_states]
        )

    def as_td_module(
        self, state_key: NestedKey, action_key: NestedKey
    ) -> TensorDictModule:
        return TensorDictModule(
            module=self,
            in_keys=[state_key],
            out_keys=[action_key],
        )
