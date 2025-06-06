from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import xmimir as xmi


@dataclass
class Plan:
    transitions: Sequence[xmi.XTransition]
    cost: float
    problem: xmi.XProblem

    def __post_init__(self):
        self.cost = sum(transition.action.cost for transition in self.transitions)


def parse_fastdownward_plan(path: Path, problem: xmi.XProblem) -> Plan:
    """
    Tries to parse plan-file by matching actions to applicable actions in the problem.
    :param path: Path to the plan file.
    :param problem: The problem for which the plan is valid.
    :return: A tuple containing a list of actions and the cost of the plan.
    """
    assert path.is_file(), path.absolute()
    lines = path.read_text().splitlines()
    succ_gen = xmi.XSuccessorGenerator(problem)
    state = succ_gen.initial_state

    # fast-downward stores plans as (action-schema obj1 obj2)
    def format_action(action: xmi.XAction):
        return f"({action.name} {' '.join(o.get_name() for o in action.objects)})"

    action_list = []
    transitions: List[xmi.XTransition] = []
    for action_name in lines:
        if not action_name.startswith("("):
            break
        action: xmi.XAction = next(
            (
                a
                for a in succ_gen.action_generator.generate_actions(state)
                if format_action(a) == action_name
            ),
            None,
        )
        if action is None:
            raise ValueError(
                "Could not find applicable action for "
                f"{action_name}. Applicable actions are"
                f"{[format_action(a) for a in succ_gen.action_generator.generate_actions(state)]}."
            )
        action_list.append(action)
        next_state = succ_gen.successor(state, action)
        transitions.append(xmi.XTransition.make_hollow(state, action, next_state))
        state = next_state
    cost = re.search(r"cost = (\d+)", lines[-1])
    if cost is None:
        raise ValueError(f"Could not find cost in {lines[-1]}")
    cost = int(cost.group(1))
    assert sum(a.cost for a in action_list) == cost
    return Plan(transitions=transitions, cost=cost, problem=problem)
