from typing import Any, Sequence

import torch
from tensordict import NestedKey, NonTensorData, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase

from rgnet.logging_setup import get_logger
from rgnet.models.atom_valuator import EmbeddingAndValuator
from rgnet.rl.envs import PlanningEnvironment
from rgnet.utils.misc import as_non_tensor_stack, tolist
from xmimir.wrappers import XAtom, XLiteral, XTransition


class SortThenCombine(TensorDictModuleBase):
    """
    Simply sorts the atoms by alphanumeric value and then follows the transitions of the first atom after sorting.

    This is used to combine the optimal goal decisions per atom to an overall goal-decision.
    """

    def __init__(
        self,
        in_keys: NestedKey | Sequence[NestedKey],
        out_keys: NestedKey | Sequence[NestedKey],
    ):
        self.in_keys = (in_keys,) if isinstance(in_keys, str) else in_keys
        self.out_keys = (out_keys,) if isinstance(out_keys, str) else out_keys
        super().__init__()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        relevant = tensordict[self.in_keys[0]]
        chosen_atom = relevant.sorted_keys[0]
        return tensordict.set(self.out_keys[0], relevant[chosen_atom])


class AtomValueActor(TensorDictModule):
    out_key = PlanningEnvironment.default_keys.action
    atom_key = "best_transitions_per_goal"

    def __init__(
        self,
        module: EmbeddingAndValuator,
        in_keys: NestedKey,
        action_key: NestedKey = out_key,
        goal_combiner: torch.nn.Module = SortThenCombine(
            in_keys=atom_key, out_keys="selected"
        ),
    ):
        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=action_key,
        )
        # module to combine the optimal goal decisions per atom to an overall goal-decision
        # (e.g. choose the most frequent best transition, or select one atom first, then another, etc...)
        self.goal_combiner = goal_combiner

    def forward(
        self,
        tensordict: TensorDictBase,
        *args,
        tensordict_out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        # we assume the tensordict has the encoded state
        states, successor_batch, transitions_stack, goals_stack = (
            tensordict.get(k) for k in self.in_keys
        )
        # unpacking `NonTensorStack`s is not fun
        transitions_list: list[list[XTransition]] = [
            ts.data for ts in transitions_stack.tensordicts
        ]
        goal_list: list[list[XLiteral]] = [gs.data for gs in goals_stack.tensordicts]
        # neither is NonTensorData
        states = states.data
        _, successor_batch = successor_batch.data
        num_successors = list(map(len, transitions_list))
        selected_td = None
        for goals, transitions, num_succ in zip(
            goal_list, transitions_list, num_successors
        ):
            atoms: list[XAtom] = [goal.atom for goal in goals]
            current_out, output_info = self.module(
                states, atoms=atoms, provide_output_metadata=True
            )
            successor_out, succ_output_info = self.module(
                successor_batch, atoms=atoms, provide_output_metadata=True
            )
            predicates = set(atom.predicate.name for atom in atoms)
            for predicate in predicates:
                current_vals = current_out[predicate].flatten()
                successor_vals = successor_out[predicate].flatten()
                num_atoms = len(current_vals)
                diff = successor_vals - current_vals.repeat(num_succ)
                # Reshape successor values into [num_succ, num_atoms] and select argmax per atom
                successor_vals = successor_vals.view(num_succ, num_atoms)
                successor_vals_masked = torch.where(
                    successor_vals < 0.1, successor_vals, -torch.inf
                )
                best_transition_per_atom = successor_vals_masked.argmax(dim=0)
                # Gather the corresponding max values
                atom_index = torch.arange(num_atoms, device=self.device)
                best_value_per_atom = successor_vals_masked[
                    best_transition_per_atom, atom_index
                ]
                # Compute per-atom improvements and gather
                diffs = diff.view(num_succ, num_atoms)
                best_value_improvement_per_atom = diffs[
                    best_transition_per_atom, atom_index
                ]
                selected_td = TensorDict(batch_size=tensordict.batch_size)
                alternatives_td = TensorDict(batch_size=tensordict.batch_size)
                for i in range(num_atoms):
                    atom = output_info[predicate][i].atom
                    selected_td[atom] = TensorDict(
                        dict(
                            transition_index=best_transition_per_atom[i].view(
                                tensordict.batch_size
                            ),
                            transition=NonTensorData(
                                transitions[best_transition_per_atom[i].item()],
                                batch_size=tensordict.batch_size,
                            ),
                            value=best_value_per_atom[i].view(tensordict.batch_size),
                            current_value=current_vals[i].view(tensordict.batch_size),
                            value_improvement=best_value_improvement_per_atom[i].view(
                                tensordict.batch_size
                            ),
                            # )
                            # )
                            # alternatives_td[atom] = TensorDict(
                            #     dict(
                            all_transitions=transitions,
                            all_values=current_vals[i].view(tensordict.batch_size),
                            all_successor_values=NonTensorData(
                                tolist(successor_vals[:, i]),
                                batch_size=tensordict.batch_size,
                            ),
                        )
                    )
        if selected_td is None:
            get_logger(__name__).warning(
                "No transitions were selected. Check that the env transforms operate as expected. Given TensorDict:\n%s",
                tensordict,
            )
            raise RuntimeError("No transitions were selected.")
        tensordict.set(
            self.atom_key,
            selected_td,
        )
        selected_td = self.goal_combiner(tensordict)
        # set the action key to the selected transition
        tensordict.set(
            self.out_keys[0],
            as_non_tensor_stack([selected_td["selected", "transition"]]),
        )
        return tensordict
