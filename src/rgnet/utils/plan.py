from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import xmimir as xmi
from xmimir import ActionHistoryDataPack, XState, XSuccessorGenerator, XTransition


@dataclass
class Plan:
    solved: bool
    transitions: Sequence[xmi.XTransition]
    problem: xmi.XProblem
    cost: float = None

    def __post_init__(self):
        self.cost = sum(transition.action.cost for transition in self.transitions)

    def str(self, detailed=False):
        if detailed:
            plan_str = (
                "[\n\t\t"
                + "\n\t\t".join([t.action.str(for_plan=True) for t in self.transitions])
                + "\n\t]"
            )
        else:
            plan_str = str(len(self.transitions))
        return (
            "\n\t".join(
                (
                    "Plan(",
                    f"solved = {self.solved},",
                    f"problem = {self.problem.name}, {self.problem.filepath},",
                    f"plan = {plan_str}{ ' (attempt)' if self.solved else ''},",
                    f"cost = {self.cost}",
                )
            )
            + f"\n)"
        )


@dataclass
class ProbabilisticPlan:
    problem: xmi.XProblem
    solved: bool
    transitions: Sequence[xmi.XTransition]
    cost: float
    average_probability: float
    min_probability: float
    rl_return: float
    subgoals: int
    cycles: Sequence[Sequence[xmi.XTransition]]
    # optimality references (optimal does not imply global optimality, but rather the best reference plan)
    optimal_transitions: Optional[Sequence[xmi.XTransition]] = None
    optimal_cost: Optional[float] = None
    # 0 if optimal, positive if higher cost than optimal
    diff_return_to_optimal: Optional[float] = None
    diff_cost_to_optimal: Optional[float] = None
    # step at which deviation from optimal plan occurred, if any and plan given
    deviation_from_optimal: Optional[int] = None

    def __post_init__(self):
        self.cost = sum(transition.action.cost for transition in self.transitions)

    # cant use dataclasses.asdict(...) because pymimir problems can't be pickled
    def serialize_as_dict(self):
        def transform(k, v):
            if isinstance(v, xmi.XProblem):
                return v.name
            elif k in ("transitions", "optimal_transitions"):
                assert isinstance(v, Sequence) and (
                    len(v) == 0 or isinstance(v[0], xmi.XTransition)
                )
                return [t.action.str(for_plan=True) for t in v]
            elif k == "cycles":
                assert isinstance(v, Sequence) and (
                    len(v) == 0
                    or (
                        isinstance(v[0], Sequence)
                        and isinstance(v[0][0], xmi.XTransition)
                    )
                )
                return [[t.action.str(for_plan=True) for t in cycle] for cycle in v]
            return v

        return {
            f.name: transform(f.name, getattr(self, f.name))
            for f in dataclasses.fields(self)
        }

    def str(self, detailed=False):
        optimal_plan_str = "/"
        if detailed:
            plan_str = (
                "[\n\t\t"
                + "\n\t\t".join([t.action.str(for_plan=True) for t in self.transitions])
                + "\n\t]"
            )
            cycle_str = (
                "[\n\t\t"
                + "\n\t\t".join(
                    [
                        "[\n\t\t\t"
                        + "\n\t\t\t".join([t.action.str(for_plan=True) for t in cycle])
                        + "\n\t\t]"
                        for cycle in self.cycles
                    ]
                )
                + "\n\t]"
            )
            if self.optimal_transitions is not None:
                optimal_plan_str = (
                    "[\n\t\t"
                    + "\n\t\t".join(
                        [t.action.str(for_plan=True) for t in self.optimal_transitions]
                    )
                    + "\n\t]"
                )
        else:
            plan_str = str(len(self.transitions))
            cycle_str = (
                "[" + ", ".join([f"{len(cycle)}" for cycle in self.cycles]) + "]"
            )
            if self.optimal_transitions is not None:
                optimal_plan_str = str(len(self.optimal_transitions))
        if self.solved:
            return (
                "\n\t".join(
                    (
                        f"{self.__class__.__name__}(",
                        f"solved = {self.solved},",
                        f"plan = {plan_str},",
                        f"RL return = {self.rl_return},",
                        f"Planning cost = {self.cost},",
                        f"subgoals = {self.subgoals},",
                        f"cycles = {cycle_str},",
                        f"average_probability = {self.average_probability},",
                        f"min_probability = {self.min_probability},",
                        f"ref. plan = {optimal_plan_str},",
                        f"return - return(ref.) = {self.diff_return_to_optimal},",
                        f"cost - cost(ref.) = {self.diff_cost_to_optimal}",
                    )
                )
                + f"\n)"
            )
        else:
            return (
                "\n\t".join(
                    (
                        f"{self.__class__.__name__}(",
                        f"solved = {self.solved},",
                        f"plan = {plan_str},",
                        f"cycles = {cycle_str},",
                        f"average_probability = {self.average_probability},",
                        f"min_probability = {self.min_probability}",
                    )
                )
                + f"\n)"
            )

    def __getstate__(self):
        # remove the problem from the state, as it cannot be pickled
        prob = self.problem
        domain_path, problem_path = prob.domain.filepath, prob.filepath
        transitions = self.transitions
        transitions = ActionHistoryDataPack(t.action for t in transitions)
        cycles = [
            ActionHistoryDataPack(t.action for t in cycle) for cycle in self.cycles
        ]
        state = self.__dict__.copy()
        state["problem"] = (domain_path, problem_path)
        state["transitions"] = transitions
        state["cycles"] = cycles
        return state

    def __setstate__(self, state):
        # restore the problem from the state
        _, state["problem"] = xmi.parse(*state["problem"])
        succ_gen = XSuccessorGenerator(state["problem"])
        state["transitions"] = state["transitions"].reconstruct_sequence(succ_gen)
        state["cycles"] = [
            cycle.reconstruct_sequence(succ_gen) for cycle in state["cycles"]
        ]
        self.__dict__.update(state)


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
                f"{[format_action(a) for a in succ_gen.action_generator.generate_actions(state)]} "
                f"in plan {path}."
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
    return Plan(transitions=transitions, cost=cost, problem=problem, solved=True)


def compute_return(gamma, transitions):
    rl_return = 0.0
    cost = 0.0
    step = 0
    for transition in transitions:
        if isinstance(transition.action, Sequence):
            rl_return += sum(
                gamma**i * (-action.cost)
                for i, action in enumerate(transition.action, start=step)
            )
            cost += sum(action.cost for action in transition.action)
            step += len(transition.action)
        else:
            rl_return += gamma**step * transition.action.cost
            cost += transition.action.cost
            step += 1
    return rl_return, cost


def analyze_cycles(transitions):
    """
    Analyze the transitions and return the cycles that were made.
    A cycle is a list of transitions that lead to a previously visited state.
    The cycles are grouped by the level of decision-making, e.g., subgoal cycles
    are on the level of subgoals, while primitive action cycles are on the level of primitive actions.
    """
    visited: set[XState] = set()
    cycles: list[list[XTransition]] = []
    current_cycle: list[XTransition] = []

    for transition in transitions:
        if transition.source in visited:
            # we have a cycle
            new_cycle = [transition]
            if len(current_cycle) > 0:
                cycles.append(current_cycle + new_cycle)
            current_cycle = new_cycle
        else:
            visited.add(transition.source)
            current_cycle.append(transition)
    return cycles
