from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import pymimir as mi

from .manual_transition import MTransition


@dataclass
class Plan:
    transitions: Sequence[mi.Transition]
    cost: float = field(init=False)  # derived from action_sequence
    problem: mi.Problem

    def __post_init__(self):
        self.cost = sum(t.action.cost for t in self.transitions)


def parse_fd_plan(path: Path, problem: mi.Problem) -> Plan:
    """
    Tries to parse plan-file by matching actions to applicable actions in the problem.
    :param path: Path to the plan file.
    :param problem: The problem for which the plan is valid.
    :return: A tuple containing a list of actions and the cost of the plan.
    """
    assert path.is_file(), path.absolute()
    lines = path.read_text().splitlines()
    succ = mi.GroundedSuccessorGenerator(problem)
    state = problem.create_state(problem.initial)

    # fast-downward stores plans as (action-schema obj1 obj2)
    def format_action(a: mi.Action):
        schema_name = a.schema.name
        obj = [o.name for o in a.get_arguments()]
        return "(" + schema_name + " " + " ".join(obj) + ")"

    action_list = []
    transitions: List[mi.Transition | MTransition] = []
    for action_name in lines:
        if not action_name.startswith("("):
            break
        action: mi.Action = next(
            (
                a
                for a in succ.get_applicable_actions(state)
                if format_action(a) == action_name
            ),
            None,
        )
        if action is None:
            raise ValueError(
                "Could not find applicable action for "
                f"{action_name}. Applicable actions are"
                f"{[format_action(a) for a in succ.get_applicable_actions(state)]}."
            )
        action_list.append(action)
        next_state = action.apply(state)
        transitions.append(MTransition(source=state, action=action, target=next_state))
        state = next_state
    cost = re.search(r"cost = (\d+)", lines[-1])
    if cost is None:
        raise ValueError(f"Could not find cost in {lines[-1]}")
    cost = int(cost.group(1))
    assert sum(a.cost for a in action_list) == cost
    return Plan(transitions=transitions, problem=problem)
