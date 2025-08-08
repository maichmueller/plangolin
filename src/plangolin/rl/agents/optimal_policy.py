from __future__ import annotations

from typing import Dict, List

from tensordict import NestedKey
from tensordict.nn import TensorDictModule

from plangolin.utils.misc import NonTensorWrapper, as_non_tensor_stack, tolist
from xmimir import XState, XStateSpace, XTransition


def optimal_action(space: XStateSpace, state: XState) -> XTransition:
    return min(
        space.forward_transitions(state),
        key=lambda t: space.goal_distance(t.target),
    )


class OptimalPolicy:
    def __init__(self, spaces: XStateSpace | List[XStateSpace]):
        spaces = [spaces] if isinstance(spaces, XStateSpace) else spaces
        self.best_actions: Dict[XState, XTransition] = {}
        for space in spaces:
            self.best_actions.update({s: optimal_action(space, s) for s in space})

    def __call__(
        self, batched_states: List[List[XState]] | NonTensorWrapper
    ) -> List[XTransition] | NonTensorWrapper:
        batched_states = tolist(batched_states)
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
