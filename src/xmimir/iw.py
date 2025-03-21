import itertools
import logging
import multiprocessing as mp
import time
from abc import abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property, singledispatchmethod
from typing import Any, Callable, Iterable, Iterator, List, NamedTuple, Sequence

import torch
from pymimir import StateSpace
from tqdm import tqdm

from .wrappers import *

__all__ = [
    "Novelty",
    "NoveltyCheck",
    "CollectorHook",
    "ExpansionNode",
    "ExpansionStrategy",
    "InOrderExpansion",
    "ReverseOrderExpansion",
    "RandomizedExpansion",
    "IWSearch",
    "IWStateSpace",
]


@dataclass(slots=True)
class NoveltyCheck:
    novelty: int
    novel_tuples: List[tuple[XAtom, ...]]

    def __bool__(self):
        return self.novelty > -1


class Novelty:
    def __init__(self, arity: int, problem: XProblem):
        self.arity = arity
        self.problem = problem
        self.known_tuples: set[tuple[XAtom, ...]] = set()

    @staticmethod
    def as_ordered_atom_tuple(atom_tuple: Iterable[XAtom]):
        return tuple(sorted(atom_tuple, key=str))

    def _tuple_generator(self, atoms: Sequence[XAtom]):
        for arity in range(1, self.arity + 1):
            for atom_tuple in itertools.combinations(atoms, arity):
                yield self.as_ordered_atom_tuple(atom_tuple)

    def add_known_tuples(self, atom_tuple_iter: Iterable[tuple[XAtom, ...]]):
        for atom_tuple in map(self.as_ordered_atom_tuple, atom_tuple_iter):
            self.known_tuples.add(atom_tuple)

    @singledispatchmethod
    def test(self, atoms: Sequence[XAtom]) -> NoveltyCheck:
        novelty = -1
        novel_tuples = []
        for atom_tuple in self._tuple_generator(atoms):
            if atom_tuple not in self.known_tuples:
                self.known_tuples.add(atom_tuple)
                novel_tuples.append(atom_tuple)
                novelty = (
                    len(novel_tuples)
                    if novelty == -1
                    else min(novelty, len(novel_tuples))
                )
        return NoveltyCheck(novelty, novel_tuples)

    @test.register
    def _(self, state: XState):
        # logically we would like a Set datastruct here with stable iteration,
        # but `set`'s iteration order is not guaranteed.
        # Hence, we convert the atom-`iterable` to a `dict` which comes with necessary guarantees.
        # Its keys are also multipass with respect to iteration and behave like a `set`
        return self.test(dict.fromkeys(state.atoms(with_statics=False)).keys())


class ExpansionNode(NamedTuple):
    state: XState
    trace: List[XTransition]
    novelty_trace: List[List[tuple[XAtom, ...]]]
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


class CollectorHook:
    def __init__(self):
        self.nodes = []

    def __call__(self, node: ExpansionNode):
        self.nodes.append(node)


class IWSearch:
    def __init__(
        self,
        width: int,
        expansion_strategy: ExpansionStrategy = InOrderExpansion(),
    ):
        self.width = width
        self.expansion_strategy = expansion_strategy
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
        goal: Sequence[XLiteral] | None = None,
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
        novelty_condition.test(start_state)
        if atom_tuples_to_avoid is not None:
            novelty_condition.add_known_tuples(atom_tuples_to_avoid)

        if goal is not None:

            def goal_check(state: XState):
                return not any(state.unsatisfied_literals(goal))

        else:
            goal_check = lambda state: state.is_goal

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
                    goal_check=goal_check,
                    novel_hook=novel_hook,
                    goal_hook=goal_hook,
                )
            )
            # clear the nodes to expand list to start a new depth
            nodes = []
            current_depth += 1

        goal_found = False
        while iteration < expansion_budget and not (
            (goal_found and stop_on_goal) or (not visit_queue and not nodes)
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
        goal_check: Callable[[XState], bool],
    ) -> list[ExpansionNode]:
        goal_nodes = []
        for state, trace, novel_sets_trace, depth in self.expansion_strategy.consume(
            nodes
        ):
            for action, child_state in successor_generator.successors(state):
                if novel_check := novelty_condition.test(child_state):
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
                    if goal_check(child_state):
                        goal_hook(child_node)
                        goal_nodes.append(child_node)
        return goal_nodes


def complete_atom_set(state_space: XStateSpace):
    all_atoms = set(state_space.initial_state.atoms(with_statics=True))
    for state in state_space:
        all_atoms.update(state.atoms(with_statics=False))
    return all_atoms


def initialize_iw_state_space_worker(
    domain_path: str,
    problem_path: str,
    space_options: dict[str, Any],
    iw: IWSearch,
    max_transitions: int,
    max_time: timedelta,
):
    globals()["state_space"] = XStateSpace(
        str(domain_path), str(problem_path), **space_options
    )
    globals()["iw"] = iw
    globals()["max_transitions"] = max_transitions
    globals()["max_time"] = max_time


def _check_timeout(
    nr_transitions: int,
    start_time: datetime,
    max_transitions: int,
    max_time: timedelta,
    problem_name: str,
):
    elapsed: timedelta = datetime.fromtimestamp(time.time()) - start_time
    if nr_transitions >= max_transitions or elapsed >= max_time:
        hours = elapsed.total_seconds() / 3600
        minutes = int((hours % 1) * 60)
        logging.info(
            f"Stopping Criterion reached for instance {problem_name}. "
            f"Transition buffer size: {nr_transitions} / {max_transitions}, "
            f"time elapsed: {hours}h {minutes}m "
            f"(max {max_time.total_seconds() / 3600}h {max_time.total_seconds() / 60}m)"
        )
        raise TimeoutError(
            f"IWStateSpace({problem_name}) construction maxed out time or transition budget."
        )


class IWStateSpace(XStateSpace):
    @dataclass(slots=True)
    class StateInfo:
        distance_to_goal: float
        distance_from_initial: float

    def __init__(
        self,
        iw: IWSearch,
        problem: str | XProblem,
        *,
        n_cpus: int = 1,
        max_transitions: int = float("inf"),
        max_time: timedelta = timedelta(hours=6),
        chunk_size: int = 100,
        **space_options,
    ):
        super().__init__(
            StateSpace.create(
                problem.domain.filepath, problem.filepath, **space_options
            )
        )
        self.iw = iw
        self.iw_fwd_transitions: dict[XState, list[XTransition]] = dict()
        self.iw_bkwd_transitions: dict[XState, list[XTransition]] = dict()
        self.state_info: dict[XState, IWStateSpace.StateInfo] = defaultdict(
            lambda: IWStateSpace.StateInfo(
                distance_to_goal=-1, distance_from_initial=-1
            )
        )
        self.space_options = space_options
        self.max_transitions = max_transitions
        self.max_time = max_time
        self.chunk_size = chunk_size
        self.n_cpus = min(n_cpus, mp.cpu_count())
        if self.n_cpus > 1:
            self._build_mp()
        else:
            self._build()

    @staticmethod
    def worker_build_transitions(state_indices):
        state_space = globals()["state_space"]
        iw = globals()["iw"]
        transitions = []
        # print("Worker started. Processing states:", state_indices)
        for idx in state_indices:
            state = state_space[idx]
            collector = CollectorHook()
            iw.solve(
                state_space.successor_generator,
                start_state=state,
                stop_on_goal=False,
                novel_hook=collector,
            )
            for node in collector.nodes:
                transitions.append((idx, len(node.trace), node.state.index))
        return transitions

    def _build_mp(self):
        start_time = datetime.fromtimestamp(time.time())
        self.iw_fwd_transitions = {state: [] for state in self}
        self.iw_bkwd_transitions = {state: [] for state in self}
        num_states = len(self)
        indices = list(range(num_states))

        num_workers = self.n_cpus
        chunks = list(
            indices[i : i + self.chunk_size]
            for i in range(0, len(indices), self.chunk_size)
        )

        # Prepare worker arguments for each small chunk.
        worker_args = (
            self.problem.domain.filepath,
            self.problem.filepath,
            self.space_options,
            self.iw,
            self.max_transitions,
            self.max_time,
        )
        all_transitions = []
        nr_transitions = 0
        with mp.Pool(
            processes=num_workers,
            initializer=initialize_iw_state_space_worker,
            initargs=worker_args,
        ) as pool:
            for result in tqdm(
                pool.imap(IWStateSpace.worker_build_transitions, chunks),
                total=len(chunks),
            ):
                all_transitions.extend(result)
                nr_transitions += len(result)
                _check_timeout(
                    nr_transitions,
                    start_time,
                    self.max_transitions,
                    self.max_time,
                    self.problem.name,
                )

        self._process_transitions(all_transitions)
        self._compute_iw_goal_distances()
        self._compute_iw_initial_distances()

    def _process_transitions(self, transitions: list[tuple[int, int, int]]):
        for source_idx, nr_actions, target_idx in transitions:
            source_state = self[source_idx]
            target_state = self[target_idx]
            transition = XTransition.make_hollow(
                source_state, [None] * nr_actions, target_state
            )
            self.iw_fwd_transitions[source_state].append(transition)
            self.iw_bkwd_transitions[target_state].append(transition)

    def _build(self):
        nr_transitions = 0
        start_time = datetime.fromtimestamp(time.time())
        for i, state in tqdm(
            enumerate(self), desc="Building IWStateSpace", total=len(self)
        ):
            state_transitions = []

            def novel_state_hook(node: ExpansionNode):
                state_transitions.append(
                    XTransition.make_hollow(
                        state, tuple(t.action for t in node.trace), node.state
                    )
                )

            self.iw.solve(
                self.successor_generator,
                start_state=state,
                novel_hook=novel_state_hook,
                stop_on_goal=False,
            )
            self.iw_fwd_transitions[state] = state_transitions
            self.iw_bkwd_transitions[state] = []  # to be filled afterwards
            nr_transitions += len(state_transitions)
            _check_timeout(
                nr_transitions,
                start_time,
                self.max_transitions,
                self.max_time,
                self.problem.name,
            )
        for bkwd_transition_list in self.iw_fwd_transitions.values():
            for transition in bkwd_transition_list:
                self.iw_bkwd_transitions[transition.target].append(transition)
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
            for transition in self.backward_transitions(current_state):
                predecessor_state = transition.source
                # logging.debug(f"Predecessor state: {predecessor_state.index}")
                # If this predecessor hasn't been assigned a goal distance yet...
                pred_state_info = self.state_info[predecessor_state]
                if pred_state_info.distance_to_goal < 0:
                    # Set its distance as current state's distance plus one.
                    pred_state_info.distance_to_goal = (
                        self.state_info[current_state].distance_to_goal + 1
                    )
                    visit_queue.append(predecessor_state)
        assert sum(is_expanded) == len(
            is_expanded
        ), f"Not all states were expanded. Indices left unexpanded: {[i for i, v in enumerate(is_expanded) if not v]}"

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
        assert sum(is_expanded) == len(is_expanded), "Not all states were expanded."

    def __len__(self):
        return len(self._vertices)

    def __str__(self):
        return (
            f"IWStateSpace("
            f"width={self.iw.width}, "
            f"#states={len(self)}, "
            f"#transitions={self.total_transition_count}, "
            f"#deadends={self.deadend_count}, "
            f"#goals={self.goal_count}, "
            f"solvable={self.solvable}, "
            f"solution_cost={self.goal_distance(self.initial_state)}"
            f")"
        )

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


if __name__ == "__main__":
    import os

    source_dir = "" if os.getcwd().endswith("/test") else "test/"
    domain_path = f"{source_dir}pddl_instances/blocks/domain.pddl"
    problem_path = f"{source_dir}pddl_instances/blocks/large.pddl"
    # problem_path = f"{source_dir}pddl_instances/blocks/iw/largish_unbound_goal.pddl"
    space = XStateSpace(domain_path, problem_path)
    iw_space = IWStateSpace(
        IWSearch(2),
        space.problem,
        n_cpus=mp.cpu_count(),
    )
