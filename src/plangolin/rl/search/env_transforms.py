from collections import defaultdict
from typing import Dict, List, Set

import torch
from tensordict import NestedKey, NonTensorStack, TensorDict, TensorDictBase
from torchrl.envs import Transform

import xmimir as xmi
from plangolin.utils.misc import as_non_tensor_stack, tolist


class CycleAvoidingTransform(Transform):
    """
    Keep track of current states and filter out any transition that leads to a visited state.
    Each batch entry is treated independently.
    """

    def __init__(self, transitions_key: NestedKey):
        super().__init__(in_keys=[transitions_key], out_keys=[transitions_key])
        self.transition_key = transitions_key
        self.visited: Dict[int, Set[xmi.XState]] = defaultdict(set)

    def _apply_transform(
        self, batched_transitions: List[List[xmi.XTransition]] | NonTensorStack
    ) -> NonTensorStack:
        batched_transitions = tolist(batched_transitions)
        filtered_transitions: list[list[xmi.XTransition]] = []
        for batch_idx, transitions in enumerate(batched_transitions):
            assert (
                len(transitions) > 0
            ), "Environment returned state without outgoing transitions."
            # all transitions should have the same source state
            self.visited[batch_idx].add(transitions[0].source)
            filtered: List[xmi.XTransition] = [
                t for t in transitions if t.target not in self.visited[batch_idx]
            ]
            filtered_transitions.append(filtered)

        return as_non_tensor_stack(filtered_transitions)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self.visited.clear()
        if tensordict.get("_reset", default=None) is not None:
            raise RuntimeError("Transform is not implemented with partial resets")
        return self._call(tensordict_reset)


class NoTransitionTruncationTransform(Transform):
    """
    After CycleAvoidingTransform has filtered out all successors,
    force a truncation (i.e. set `truncated=True` and thus `done=True`)
    on any batch index where `transitions` is empty.
    """

    def __init__(
        self,
        transitions_key: NestedKey,
        done_key: NestedKey,
        truncated_key: NestedKey,
    ):
        # We do NOT pass anything to in_keys / out_keys, because we override _step directly.
        super().__init__(
            in_keys=None, out_keys=None, in_keys_inv=None, out_keys_inv=None
        )
        self.transitions_key = transitions_key
        self.done_key = done_key
        self.truncated_key = truncated_key

    def _step(
        self,
        tensordict: TensorDict,  # the TensorDict at time t
        next_tensordict: TensorDict,  # the TensorDict at time t+1 (after env.step)
    ) -> TensorDict:
        """
        Called after the base env (and any previous transforms) have stepped.
        We look at `next_tensordict[self.transitions_key]` and, wherever that list is empty,
        we force `truncated=True` and then `done=True`.
        """
        batched_transitions = next_tensordict.get(self.transitions_key)
        transitions_list = tolist(batched_transitions)  # now a Python list of lists

        batch_size = len(transitions_list)
        device = next_tensordict.device

        truncated = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        for i in range(batch_size):
            if len(transitions_list[i]) == 0:
                truncated[i, 0] = True

        # read existing `done` (if absent, assume all False)
        if next_tensordict.get(self.done_key, None) is None:
            done_existing = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=device
            )
        else:
            done_existing = next_tensordict.get(self.done_key)

        new_done = done_existing | truncated
        next_tensordict.set(self.truncated_key, truncated)
        next_tensordict.set(self.done_key, new_done)

        return next_tensordict
