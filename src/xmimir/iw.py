import itertools
import logging
import time
from abc import abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import cached_property, singledispatchmethod
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple

import torch
from pymimir import StateSpace

from .wrappers import *


class NoveltyCheck(NamedTuple):
    is_novel: bool
    novel_tuples: List[tuple[XAtom, ...]]


class Novelty:
    def __init__(self, arity: int, problem: XProblem):
        self.arity = arity
        self.problem = problem
        self.visited_lookup: Dict[tuple[XAtom, ...], bool] = defaultdict(bool)

    @staticmethod
    def ordered_atom_tuple(atom_tuple: Iterable[XAtom]):
        return tuple(sorted(atom_tuple, key=hash))

    def _tuple_generator(self, atoms: Iterable[XAtom]):
        for arity in range(1, self.arity + 1):
            for atom_tuples in itertools.combinations_with_replacement(atoms, arity):
                atom_set = set(atom_tuples)
                if len(atom_set) < arity:
                    # we have yielded this atom already in a previous iteration
                    continue
                yield self.ordered_atom_tuple(atom_set)

    def mark_visited(self, atom_tuple_iter: Iterable[tuple[XAtom, ...]]):
        for atom_tuple in map(self.ordered_atom_tuple, atom_tuple_iter):
            self.visited_lookup[atom_tuple] = True

    @singledispatchmethod
    def test(self, atoms: Iterable[XAtom]) -> NoveltyCheck:
        novel = False
        novel_tuples = []
        for atom_tuple in self._tuple_generator(atoms):
            if not self.visited_lookup[atom_tuple]:
                self.visited_lookup[atom_tuple] = True
                novel_tuples.append(atom_tuple)
                novel = True
        return NoveltyCheck(novel, novel_tuples)

    @test.register
    def _(self, state: XState):
        return self.test(state.atoms(with_statics=False))


class ExpansionNode(NamedTuple):
    state: XState
    trace: List[XTransition]
    novelty_trace: List[tuple[XAtom, ...]]
    depth: int


class ExpansionStrategy:
    def __init__(self):
        self.options = None

    def consume(self, options: List[ExpansionNode]):
        self.options = options
        return self

    @abstractmethod
    def __iter__(self) -> Iterator[ExpansionNode]: ...


class InOrderExpansion(ExpansionStrategy):
    def __iter__(self):
        return iter(self.options)


class ReverseOrderExpansion(ExpansionStrategy):
    def __iter__(self):
        return reversed(self.options)


class RandomizedExpansion(ExpansionStrategy):
    def __init__(self, seed: int):
        super().__init__()
        self.rng = torch.random.manual_seed(seed)

    def __iter__(self):
        for index in torch.randperm(len(self.options), generator=self.rng):
            yield self.options[index]


class IWSearch:
    def __init__(
        self,
        width: int,
        expansion_strategy: ExpansionStrategy = InOrderExpansion(),
    ):
        self.width = width
        self.expansion_strat = expansion_strategy
        self.current_novelty_condition: Novelty | None = None
        self.current_successor_generator = None

    @property
    def current_problem(self):
        return self.current_successor_generator.problem

    def solve(
        self,
        successor_generator: XSuccessorGenerator,
        novelty_condition: Novelty | None = None,
        start_state: XState | None = None,
        stop_on_goal: bool = True,
        atom_tuples_to_avoid: Iterable[tuple[XAtom, ...]] | None = None,
        novel_hook: Callable[[ExpansionNode], None] = lambda *a, **kw: ...,
        goal_hook: Callable[[ExpansionNode], None] = lambda *a, **kw: ...,
        expansion_budget: int = float("inf"),
    ) -> List[XTransition] | None:
        self.current_successor_generator = successor_generator
        if novelty_condition is None:
            novelty_condition = Novelty(self.width, self.current_problem)
        self.current_novelty_condition = novelty_condition

        if start_state is None:
            start_state = successor_generator.initial_state
        novelty_condition.test(start_state.atoms(with_statics=False))
        if atom_tuples_to_avoid is not None:
            novelty_condition.mark_visited(atom_tuples_to_avoid)

        visit_queue: deque[ExpansionNode] = deque(
            [ExpansionNode(start_state, [], [], 0)]
        )
        goal_traces = []
        iteration = 0
        current_depth = 0
        nodes: List[ExpansionNode] = []

        def process_nodes():
            """
            Process the nodes to expand at the current depth level.
            """
            nonlocal nodes, current_depth
            goal_traces.extend(
                self._process_nodes(
                    nodes,
                    visit_queue,
                    novelty_condition,
                    successor_generator,
                    novel_hook=novel_hook,
                    goal_hook=goal_hook,
                )
            )
            # clear the nodes to expand list to start a new depth
            nodes = []
            current_depth += 1

        goal_found = False
        while iteration < expansion_budget and not (
            (goal_found and stop_on_goal) or (visit_queue and not nodes)
        ):
            if not visit_queue:
                process_nodes()
            else:
                elem = visit_queue.popleft()
                if elem.depth > current_depth:
                    process_nodes()

                nodes.append(elem)

            iteration += 1
        return goal_traces

    def _process_nodes(
        self,
        nodes: List[ExpansionNode],
        visit_queue: deque[ExpansionNode],
        novelty_condition: Novelty,
        successor_generator: XSuccessorGenerator,
        novel_hook: Callable[[ExpansionNode], None],
        goal_hook: Callable[[ExpansionNode], None],
    ) -> list[ExpansionNode]:
        goal_nodes = []
        for state, trace, novel_sets_trace, depth in self.expansion_strat.consume(
            nodes
        ):
            for action, child_state in successor_generator.successors(state):
                if (novel_check := novelty_condition.test(child_state)).is_novel:
                    child_trace = trace + [
                        XTransition.make_hollow(state, action, child_state)
                    ]
                    child_novel_sets_trace = novel_sets_trace + [
                        novel_check.novel_tuples
                    ]
                    child_node = ExpansionNode(
                        child_state, child_trace, child_novel_sets_trace, depth + 1
                    )
                    novel_hook(child_node)
                    visit_queue.append(child_node)
                    if child_state.is_goal:
                        goal_hook(child_node)
                        goal_nodes.append(child_node)
        return goal_nodes


def atom_set(state_space: XStateSpace):
    all_atoms = set(state_space.initial_state.atoms(with_statics=True))
    for state in state_space:
        all_atoms.update(state.atoms(with_statics=False))
    return all_atoms


class IWStateSpace(XStateSpace):
    class StateInfo(NamedTuple):
        distance_to_goal: float
        distance_from_initial: float

    def __init__(
        self,
        iw: IWSearch,
        primitive_space: XStateSpace | StateSpace,
        *,
        max_transitions: int = float("inf"),
        max_time: timedelta = timedelta(hours=6),
    ):
        super().__init__(
            primitive_space
            if isinstance(primitive_space, StateSpace)
            else primitive_space.base
        )
        self.iw = iw
        self.iw_fwd_transitions: dict[XState, list[XTransition]] = dict()
        self.iw_bkwd_transitions: dict[XState, list[XTransition]] = dict()
        self.state_info: dict[XState, IWStateSpace.StateInfo] = defaultdict(
            lambda: IWStateSpace.StateInfo(
                distance_to_goal=-1, distance_from_initial=-1
            )
        )
        self.max_transitions = max_transitions
        self.max_time = max_time

    def _build(self):
        nr_transitions = 0
        start_time = datetime.fromtimestamp(time.time())
        for state in self:
            state_fwd_transitions = []
            state_bkwd_transitions = []

            def novel_state_hook(node: ExpansionNode):
                state_fwd_transitions.append(
                    XTransition.make_hollow(
                        state, tuple(t.action for t in node.trace), node.state
                    )
                )
                state_bkwd_transitions.append(
                    XTransition.make_hollow(state, [None] * len(node.trace), node.state)
                )

            self.iw.solve(
                self.successor_generator,
                start_state=state,
                novel_hook=novel_state_hook,
                stop_on_goal=False,
            )
            self.iw_fwd_transitions[state] = state_fwd_transitions
            self.iw_bkwd_transitions[state] = state_bkwd_transitions
            nr_transitions += len(state_fwd_transitions)

            elapsed: timedelta = datetime.fromtimestamp(time.time()) - start_time
            if nr_transitions >= self.max_transitions or elapsed >= self.max_time:
                hours = elapsed.total_seconds() / 3600
                minutes = int((hours % 1) * 60)
                logging.info(
                    f"Stopping Criterion reached for instance {self.problem.name}. "
                    f"Transition buffer size: {nr_transitions} / {self.max_transitions}, "
                    f"time elapsed: {hours}h {minutes}m "
                    f"(max {self.max_time.total_seconds() / 3600}h {self.max_time.total_seconds() / 60}m)"
                )
                return None

        self._compute_iw_goal_distances()
        self._compute_iw_initial_distances()

    def _compute_iw_goal_distances(self):
        """
        Computes the distance from each state to a goal state by doing a backward BFS
        starting from all goal states.
        """
        # Create a deque for the states to visit.
        visit_queue = deque()

        for goal_state in self.goal_states_iter():
            self.state_info[goal_state].distance_to_goal = 0
            visit_queue.append(goal_state)

        # A list to track which states have already been expanded.
        is_expanded = [False] * len(self)

        # Process the queue until empty.
        while visit_queue:
            current_state = visit_queue.popleft()
            if is_expanded[current_state.index]:
                continue
            is_expanded[current_state.index] = True

            # Debug log for current state.
            # logging.debug(
            #     f"Current state: {current_state} "
            #     f"(is goal: {current_state.is_goal}, "
            #     f"is initial: {self.successor_generator.initial_state == current_state}), "
            #     f"Number of predecessors: {self.backward_transition_count(current_state)}"
            # )
            # For each predecessor (using backward transitions)
            for transition in self.backward_transitions(current_state):
                predecessor_state = transition.target
                # logging.debug(f"Predecessor state: {predecessor_state.index}")
                # If this predecessor hasn't been assigned a goal distance yet...
                if (
                    pred_state_info := self.state_info[predecessor_state]
                ).distance_to_goal < 0:
                    # Set its distance as current state's distance plus one.
                    pred_state_info.distance_to_goal = (
                        self.state_info[current_state].distance_to_goal + 1
                    )
                    visit_queue.append(predecessor_state)

    def _compute_iw_initial_distances(self):
        """
        Computes the distance from the initial state to every other state using
        a forward BFS. The computed distances are stored in the 'distance_from_initial'
        field of the state_info dictionary.
        """
        # Set the initial state's distance-from-initial to 0.
        initial_state = self.initial_state  # or use self.get_initial_state() if defined
        self.state_info[initial_state] = self.StateInfo(
            distance_to_goal=self.state_info[initial_state].distance_to_goal,
            distance_from_initial=0,
        )

        visit_queue = deque([initial_state])
        # Create a boolean list to mark whether a state (by index) has been expanded.
        is_expanded = [False] * len(self)

        while visit_queue:
            current_state = visit_queue.popleft()
            if is_expanded[current_state.index]:
                continue
            is_expanded[current_state.index] = True

            # logging.debug(
            #     f"Current state: {current_state.index} (is goal: {current_state.is_goal}, "
            #     f"is initial: {self.initial_state == current_state}), "
            #     f"number of successors: {self.forward_transition_count(current_state)}"
            # )

            for transition in self.forward_transitions(current_state):
                successor_state = transition.target
                # if the successor's distance-from-initial is not yet set (< 0), update it.
                if (
                    successor_state_info := self.state_info[successor_state]
                ).distance_from_initial < 0:
                    successor_state_info.distance_from_initial = (
                        self.state_info[current_state].distance_from_initial + 1
                    )
                    visit_queue.append(successor_state)

    def __len__(self):
        return len(self._vertices)

    def __str__(self):
        return (
            f"IWStateSpace("
            f"width={self.iw.width}, "
            f"#states={len(self)}, "
            f"#transitions={self.total_transition_count}), "
            f"#deadends={self.deadend_count}, "
            f"#goals={self.goal_count}, "
            f"solvable={self.solvable}, "
            f"solution_cost={self.goal_distance(self.initial_state)}"
        )

    def str(self):
        return f"IW{self.iw.width}StateSpace({str(self.base)})"

    @cached_property
    def total_transition_count(self) -> int:
        return sum(len(transitions) for transitions in self.iw_fwd_transitions.values())

    def goal_distance(self, state: XState) -> float:
        return self.state_info[state].distance_to_goal

    def forward_transitions(self, state: XState) -> Iterator[XTransition]:
        return iter(self.iw_fwd_transitions[state])

    def backward_transitions(self, state: XState) -> Iterator[XTransition]:
        return iter(self.iw_bkwd_transitions[state])

    def _transitions(self, state: XState, direction: str) -> Iterator[XTransition]:
        if direction == "forward":
            return self.forward_transitions(state)
        else:
            return self.backward_transitions(state)

    def _transition_count(self, state: XState, direction: str) -> int:
        if direction == "forward":
            return len(self.iw_fwd_transitions[state])
        else:
            return len(self.iw_bkwd_transitions[state])
