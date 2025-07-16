import itertools
import time
from abc import abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property, singledispatchmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Sequence,
)

import torch
from multimethod import multimethod

from rgnet.logging_setup import get_logger, tqdm
from rgnet.utils.misc import KeyAwareDefaultDict, env_aware_cpu_count, return_true

# from .extensions import *
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
    "IWSuccessorGenerator",
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

    def _tuple_generator(self, atoms: Sequence[XAtom], added_atoms: Sequence[XAtom]):
        """
        Generates ordered tuples of atoms of the given arity from the provided atom set.

        Note: If added_atoms is not provided, the entire atom set is used.
              We expect added_atoms to be a subset of atoms.
        """
        for arity in range(1, self.arity + 1):
            match arity:
                case 1:
                    for atom in added_atoms or atoms:
                        yield self.as_ordered_atom_tuple((atom,))
                case 2:
                    for a1 in added_atoms or atoms:
                        for a2 in atoms:
                            yield self.as_ordered_atom_tuple((a1, a2))
                case _:
                    for atom_tuple in itertools.combinations(atoms, arity):
                        yield self.as_ordered_atom_tuple(atom_tuple)

    def add_known_tuples(self, atom_tuple_iter: Iterable[tuple[XAtom, ...]]):
        for atom_tuple in map(self.as_ordered_atom_tuple, atom_tuple_iter):
            self.known_tuples.add(atom_tuple)

    @singledispatchmethod
    def test(
        self, atoms: Sequence[XAtom], added_atoms: Sequence[XAtom] = tuple()
    ) -> NoveltyCheck:
        novelty = -1
        novel_tuples = []
        for atom_tuple in self._tuple_generator(atoms, added_atoms):
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
    def _(self, state: XState, added_atoms: Sequence[XAtom] = tuple()) -> NoveltyCheck:
        # logically we would like a Set datastruct here with stable iteration,
        # but `set`'s iteration order is not guaranteed.
        # Hence, we convert the atom-`iterable` to a `dict` which comes with necessary guarantees.
        # Its keys are also multipass with respect to iteration and behave like a `set`
        return self.test(
            dict.fromkeys(state.atoms(with_statics=False)).keys(),
            added_atoms,
        )


class ExpansionNode(NamedTuple):
    state: XState
    trace: List[XTransition]
    novelty_trace: List[List[tuple[XAtom, ...]]]
    depth: int
    effective_width: int


class ExpansionStrategy:
    def __init__(self):
        self.options: list[ExpansionNode] | None = None
        self._start_state: XState | None = None

    @property
    def start_state(self) -> XState:
        return self._start_state

    @start_state.setter
    def start_state(self, state: XState):
        self._start_state = state

    def consume(self, options: List[ExpansionNode]):
        self.options = options
        return self

    def __eq__(self, other):
        return self.__class__ == other.__class__

    @abstractmethod
    def __iter__(self) -> Iterator[ExpansionNode]: ...

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __getstate__(self):
        """
        Custom serialization to avoid issues with unhashable types in options.
        """
        state = self.__dict__.copy()
        state["options"] = None
        state["_start_state"] = None
        return state

    def __setstate__(self, state):
        """
        Custom deserialization to avoid issues with unhashable types in options.
        """
        self.__dict__.update(state)


class InOrderExpansion(ExpansionStrategy):
    def __iter__(self):
        return iter(self.options)


class ReverseOrderExpansion(ExpansionStrategy):
    def __iter__(self):
        return reversed(self.options)


class RandomizedExpansion(ExpansionStrategy):
    def __init__(self, seed: int | None = None):
        super().__init__()
        self.seed = seed
        if seed is None:
            seed = torch.seed()
        self.rng = torch.random.manual_seed(seed)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.seed == other.seed

    def __iter__(self):
        for index in torch.randperm(len(self.options), generator=self.rng):
            yield self.options[index]

    def __str__(self):
        return f"{self.__class__.__name__}(seed={self.seed})"

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("rng", None)  # remove the unhashable rng
        if self.seed is not None:
            state["rng_seed"] = self.rng.initial_seed()
            # convert the ByteTensor into a plain Python list of ints (otherwise bizarre pickling errors occur with `spawn`)
            state["rng_state_list"] = self.rng.get_state().tolist()
        state["seed"] = self.seed
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.seed = state["seed"]
        gen = torch.Generator()
        if self.seed is not None:
            # set the seed for the generator
            gen.manual_seed(state["rng_seed"])
            # rebuild a ByteTensor from the Python list
            gen.set_state(torch.ByteTensor(state["rng_state_list"]))
        else:
            gen = gen.manual_seed(gen.seed())
        self.rng = gen


class GoalProgressionExpansion(ExpansionStrategy):
    """
    An expansion strategy that expands nodes based whether they do not undo already satisfied goals.
    Nodes that maintain the goal progression are expanded first.
    """

    def __init__(self):
        super().__init__()
        self._goal: tuple[XLiteral, ...] | None = None
        self._nr_goals: int | None = None
        self._satisfied_goals: set[XLiteral] | None = None

    @property
    def start_state(self) -> XState | None:
        return self._start_state

    @start_state.setter
    def start_state(self, state: XState):
        """
        Sets the start state for the expansion strategy.
        This is used to determine the goal literals that should not be undone.
        """
        self._start_state = state
        self._goal = state.problem.goal()
        self._nr_goals = len(self._goal)
        self._satisfied_goals = set(state.satisfied_literals(self._goal))

    def consume(self, options: List[ExpansionNode]):
        """
        Consumes the options and sets the goal for the expansion strategy.
        This is used to determine which nodes should be expanded first based on goal satisfaction.
        """
        # sort in descending order, the higher the number the better
        options = sorted(options, key=self._goal_satisfaction_order, reverse=True)
        self.options = options
        return self

    def _goal_satisfaction_order(self, node: ExpansionNode) -> int:
        """
        Returns the number of goal literals that are satisfied by the node's state.
        This is used to determine the order in which nodes should be expanded.
        """
        node_satisfied_literals = set(node.state.satisfied_literals(self._goal))
        undone_goals = self._satisfied_goals - node_satisfied_literals
        newly_satisfied_goals = node_satisfied_literals - self._satisfied_goals
        # Ranking by undone goals first, and new goals added second.
        # Since #new_goals <= #goals each undone goal lowers the score to a new plateau
        # that cannot be surpassed by new goals added, hence ensuring the factored ranking.
        return -(len(undone_goals) * self._nr_goals) + len(newly_satisfied_goals)

    def __iter__(self):
        return iter(self.options)

    def __getstate__(self):
        """
        Custom serialization to avoid issues with unhashable types in options.
        """
        state = super().__getstate__()
        state["_goal"] = None
        state["_nr_goals"] = None
        state["_satisfied_goals"] = None
        return state

    def __setstate__(self, state):
        """
        Custom deserialization to avoid issues with unhashable types in options.
        """
        super().__setstate__(state)
        self._goal = None
        self._nr_goals = None
        self._satisfied_goals = None


class PreferentialExpansion(InOrderExpansion):
    """
    An expansion strategy that expands nodes in-order, but selects only specific nodes at a specific depths.
    """

    def __init__(self, depth: int, node_selector: Callable[[ExpansionNode], bool]):
        """
        :param depth: The depth at which to select nodes.
        :param node_selector: A callable that takes an ExpansionNode and returns True if the node should be selected.
        """
        super().__init__()
        self.depth = depth
        self.node_selector = node_selector

    def consume(self, options: List[ExpansionNode]):
        self.options = [
            node
            for node in options
            if node.depth == self.depth and self.node_selector(node)
        ]
        return self


class CollectorHook:
    def __init__(self):
        self.nodes = []

    def __call__(self, node: ExpansionNode):
        self.nodes.append(node)


class IWSearch:
    """
    Performs an Iterative Width Search IW(k) of a specific expansion strategy and width k.

    Note that we follow the more recent definition of IW(k) as in the paper:
        https://www.jair.org/index.php/jair/article/view/15581
    """

    def __init__(
        self,
        width: int,
        expansion_strategy: ExpansionStrategy = InOrderExpansion(),
        depth_1_is_novel: bool = True,
    ):
        """
        :param width: The width of the search.
        :param expansion_strategy: The strategy to use for expanding nodes.
        :param depth_1_is_novel: If True, every state at the first depth is always considered novel.
        """
        self.width = width
        self.expansion_strategy = expansion_strategy
        self.depth_1_is_novel: bool = depth_1_is_novel and width > 0

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.width == other.width
            and self.expansion_strategy == other.expansion_strategy
            and self.depth_1_is_novel == other.depth_1_is_novel
        )

    def __str__(self):
        return f"IWSearch(width={self.width}, expansion_strategy={self.expansion_strategy}, depth_1_is_novel={self.depth_1_is_novel})"

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
        expansion_budget: Callable[[int], bool] = return_true,
    ) -> List[ExpansionNode]:
        if novelty_condition is None:
            novelty_condition = Novelty(self.width, successor_generator.problem)

        if start_state is None:
            start_state = successor_generator.initial_state
        # reset the start state for the expansion strategy
        self.expansion_strategy.start_state = start_state

        novelty_condition.test(start_state)
        if atom_tuples_to_avoid is not None:
            novelty_condition.add_known_tuples(atom_tuples_to_avoid)

        if goal is not None:

            def goal_check(state: XState):
                return not any(state.unsatisfied_literals(goal))

        else:
            goal_check = lambda state: state.is_goal()

        visit_queue: deque[ExpansionNode] = deque(
            [ExpansionNode(start_state, [], [], 0, 0)]
        )
        start_is_goal = goal_check(start_state)
        if start_is_goal:
            goal_traces = [visit_queue[0]]
            if stop_on_goal:
                return goal_traces
        else:
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
                    stop_on_goal=stop_on_goal,
                    goal_check=goal_check,
                    novel_hook=novel_hook,
                    goal_hook=goal_hook,
                )
            )
            # clear the nodes to expand list to start a new depth
            nodes = []
            current_depth += 1

        goal_found = False
        while expansion_budget(iteration) and not (
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
        self.expansion_strategy.options.clear()
        return goal_traces

    def _process_nodes(
        self,
        nodes: List[ExpansionNode],
        visit_queue: deque[ExpansionNode],
        novelty_condition: Novelty,
        successor_generator: XSuccessorGenerator,
        stop_on_goal: bool,
        novel_hook: Callable[[ExpansionNode], None],
        goal_hook: Callable[[ExpansionNode], None],
        goal_check: Callable[[XState], bool],
    ) -> list[ExpansionNode]:
        goal_nodes = []
        for (
            state,
            trace,
            novel_sets_trace,
            depth,
            effective_width,
        ) in self.expansion_strategy.consume(nodes):
            for _, child_state, action in successor_generator.successors(state):
                pos_effect_atoms = tuple(action.effects(positive=True))
                if not pos_effect_atoms:
                    continue
                novel_check = novelty_condition.test(child_state, pos_effect_atoms)
                is_novel = novel_check or (depth == 0 and self.depth_1_is_novel)
                is_goal = goal_check(child_state)
                if not (is_goal or is_novel):
                    continue
                if effective_width == -1:
                    # we are already on a path of width potentially higher than k
                    child_novelty = -1
                elif depth == 0 and self.depth_1_is_novel and not novel_check:
                    # override the child novelty, since the actual child's novelty is higher than our width,
                    # so we cannot guarantee that this path has width <= k
                    child_novelty = -1
                else:
                    child_novelty = max(effective_width, novel_check.novelty)
                child_node = ExpansionNode(
                    child_state,
                    trace + [XTransition.make_hollow(state, action, child_state)],
                    novel_sets_trace + [novel_check.novel_tuples],
                    depth + 1,
                    child_novelty,
                )
                if is_novel:
                    novel_hook(child_node)
                    visit_queue.append(child_node)
                if is_goal:
                    goal_hook(child_node)
                    goal_nodes.append(child_node)
                    if stop_on_goal:
                        return goal_nodes
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
        get_logger(__name__).info(
            f"Stopping Criterion reached for instance {problem_name}. "
            f"Transition buffer size: {nr_transitions} / {max_transitions}, "
            f"time elapsed: {hours}h {minutes}m "
            f"(max {max_time.total_seconds() / 3600}h {max_time.total_seconds() / 60}m)"
        )
        raise TimeoutError(
            f"IWStateSpace({problem_name}) construction maxed out time or transition budget."
        )


class IWSuccessorGenerator(XSuccessorGenerator):
    """
    A successor generator that uses an IWSearch to generate successors for a given state.
    """

    @multimethod
    def __init__(self, iw_search: IWSearch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iw_search = iw_search

    @multimethod
    def __init__(
        self,
        iw_search: IWSearch,
        successor_generator: XSuccessorGenerator,
    ):
        grounder = successor_generator.grounder
        action_generator = successor_generator.action_generator
        state_repository = successor_generator.base
        super().__init__(grounder, state_repository, action_generator)
        self.iw_search = iw_search

    def successor(self, state: XState, action: XAction | Sequence[XAction]) -> XState:
        if isinstance(action, XAction):
            return super().successor(state, action)
        elif isinstance(action, Sequence):
            state = state
            for act in action:
                state = super().successor(state, act)
            return state
        else:
            raise TypeError(
                f"Action must be an XAction or a sequence of XActions, got {type(action)}"
            )

    def successors(self, state: XState) -> Iterator[XTransition]:
        collector = CollectorHook()
        self.iw_search.solve(
            self,
            start_state=state,
            stop_on_goal=False,
            novel_hook=collector,
        )
        for node in collector.nodes:
            trace = node.trace
            yield XTransition.make_hollow(
                state, [tr.action for tr in trace], node.state
            )

    __hash__ = None


class IWStateSpace(XStateSpace):
    @dataclass(slots=True)
    class StateInfo:
        distance_to_goal: float
        distance_from_initial: float

    @multimethod
    def __init__(self, iw: IWSearch, space: XStateSpace, **kwargs):
        super().__init__(space.base)
        self._init(iw, **kwargs)

    @multimethod
    def __init__(self, iw: IWSearch, problem: XProblem, **kwargs):
        super().__init__(problem)
        self._init(iw, **kwargs)

    @multimethod
    def __init__(
        self, iw: IWSearch, domain_path: Path | str, problem_path: Path | str, **kwargs
    ):
        super().__init__(domain_path, problem_path)
        self._init(iw, **kwargs)

    def _init(
        self,
        iw: IWSearch,
        *,
        serialized_transitions: list[tuple[int, tuple[int, ...], int]] | None = None,
        serialized_state_infos: list[StateInfo] | None = None,
        n_cpus: int | str = 1,
        max_transitions: int = float("inf"),
        max_time: timedelta = timedelta(hours=6),
        chunk_size: int = 100,
        pbar: bool = True,
        **space_options,
    ):
        self.iw = iw
        self.iw_fwd_transitions: dict[XState, list[XTransition]] = dict()
        self.iw_bkwd_transitions: dict[XState, list[XTransition]] = dict()
        self._serialized_transitions: list[tuple[int, tuple[int, ...], int]] = (
            serialized_transitions or []
        )
        self.space_options = space_options
        self.max_transitions = max_transitions
        self.max_time = max_time
        self.chunk_size = chunk_size
        self.pbar = pbar
        self.n_cpus = (
            min(n_cpus, env_aware_cpu_count())
            if n_cpus != "auto"
            else env_aware_cpu_count()
        )
        if self._serialized_transitions:
            for state in self:
                self.iw_fwd_transitions[state] = []
                self.iw_bkwd_transitions[state] = []
            self._load_from_serialized_transitions(self._serialized_transitions)
        else:
            if self.n_cpus > 1:
                self._build_mp()
            else:
                self._build()
        self.state_info: dict[XState, IWStateSpace.StateInfo]
        if serialized_state_infos:
            self.state_info = {
                self[state]: info for state, info in enumerate(serialized_state_infos)
            }
        else:
            self.state_info = defaultdict(
                lambda: IWStateSpace.StateInfo(
                    distance_to_goal=-1, distance_from_initial=-1
                )
            )
            self._compute_iw_goal_distances()
            self._compute_iw_initial_distances()

    @property
    def serialized_transitions(self):
        if not self._serialized_transitions:
            self.serialize_transitions()
        return self._serialized_transitions

    @staticmethod
    def worker_build_transitions(state_indices):
        state_space = globals()["state_space"]
        iw = globals()["iw"]
        transitions = []
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
                transitions.append(
                    (idx, tuple(t.action.index for t in node.trace), node.state.index)
                )
        return transitions

    def _build_mp(self):
        if not Path(self.problem.domain.filepath).exists():
            raise FileNotFoundError(
                f"Domain file {self.problem.domain.filepath} does not exist."
            )
        if not Path(self.problem.filepath).exists():
            raise FileNotFoundError(
                f"Problem file {self.problem.filepath} does not exist."
            )
        start_time = datetime.fromtimestamp(time.time())
        num_states = len(self)
        indices = list(range(num_states))

        chunks = list(
            indices[i : i + self.chunk_size]
            for i in range(0, len(indices), self.chunk_size)
        )
        if len(chunks) == 1:
            self._build()
            return

        num_workers = min(env_aware_cpu_count(), self.n_cpus, len(chunks))
        for state in self:
            self.iw_fwd_transitions[state] = []
            self.iw_bkwd_transitions[state] = []

        # Prepare worker arguments for each small chunk.
        worker_args = (
            self.problem.domain.filepath,
            self.problem.filepath,
            self.space_options,
            self.iw,
            self.max_transitions,
            self.max_time,
        )
        nr_transitions = 0
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=initialize_iw_state_space_worker,
            initargs=worker_args,
        ) as executor:
            iterable = executor.map(IWStateSpace.worker_build_transitions, chunks)
            progress_iterable = (
                tqdm(
                    iterable,
                    total=len(chunks),
                    desc=f"Building IWStateSpace({Path(self.problem.filepath).stem}) - #Proc: {num_workers}",
                    ncols=80 + len(f" - #Proc: {num_workers}"),
                )
                if self.pbar
                else iterable
            )
            for result in progress_iterable:
                self._serialized_transitions.extend(result)
                self._load_from_serialized_transitions(result)
                nr_transitions += len(result)
                _check_timeout(
                    nr_transitions,
                    start_time,
                    self.max_transitions,
                    self.max_time,
                    self.problem.name,
                )

    def _load_from_serialized_transitions(
        self, transitions: list[tuple[int, tuple[int, ...], int]]
    ):
        action_gen = self.successor_generator.action_generator
        for source_idx, action_indices, target_idx in transitions:
            source_state = self[source_idx]
            target_state = self[target_idx]
            transition = XTransition.make_hollow(
                source_state,
                [action_gen.get_action(idx) for idx in action_indices],
                target_state,
            )
            self.iw_fwd_transitions[source_state].append(transition)
            self.iw_bkwd_transitions[target_state].append(transition)

    def _build(self):
        nr_transitions = 0
        start_time = datetime.fromtimestamp(time.time())
        iterable = enumerate(self)
        for i, state in (
            pbar := (
                tqdm(
                    iterable,
                    desc=f"Building IWStateSpace({Path(self.problem.filepath).stem})",
                    total=len(self),
                )
                if self.pbar
                else iterable
            )
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

            for transition in self.backward_transitions(current_state):
                predecessor_state = transition.source
                # If this predecessor hasn't been assigned a goal distance yet...
                pred_state_info = self.state_info[predecessor_state]
                if pred_state_info.distance_to_goal < 0:
                    # Set its distance as current state's distance plus one.
                    pred_state_info.distance_to_goal = (
                        self.state_info[current_state].distance_to_goal + 1
                    )
                    visit_queue.append(predecessor_state)
        assert sum(is_expanded) == (
            len(self) - self.deadend_count
        ), f"Not all non-deadend states were expanded. Indices left unexpanded: {[i for i, v in enumerate(is_expanded) if not v]}"

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
        assert sum(is_expanded) == len(
            is_expanded
        ), f"Not all states were expanded. Indices left unexpanded: {[i for i, v in enumerate(is_expanded) if not v]}"

    def a_star_search(
        self,
        target: XState,
        start: XState | None = None,
        heuristic: Mapping[int, float] | None = None,
        forward: bool = True,
    ) -> list[XTransition]:
        """
        Performs a forward A* search from the initial state to the target state.
        :param target: The target state to reach.
        :param start: The starting state, defaults to the initial state of the space.
        :param heuristic: Precomputed distances from the start state to each state.
        :param forward: If True, performs a forward search; otherwise, a backward search.
        :return: A list of transitions leading to the target state.
        """
        if start is None:
            start = self.initial_state
        if heuristic is None:
            if start.semantic_eq(self.initial_state):
                # If the start state is the initial state, we can use the precomputed distances.
                heuristic = KeyAwareDefaultDict(
                    lambda index: self.state_info[self[index]].distance_from_initial
                )
            else:
                # otherwise, we have no available distances, so we error out.
                raise ValueError(
                    "No distances from the start state are available. "
                    "Please provide dists_from_start or ensure the start state is the initial state."
                )
        return super().a_star_search(
            target=target, start=start, heuristic=heuristic, forward=forward
        )

    def shortest_forward_distances_from_state(self, state: int | XState) -> list[float]:
        get_logger(__name__).warning(
            "The method `shortest_forward_distances_from_state` uses primitive action distances."
        )
        return super().shortest_forward_distances_from_state(state)

    def __getstate__(self):
        infos = [None] * len(self.state_info)
        for s, info in self.state_info.items():
            infos[s.index] = info
        if not self._serialized_transitions:
            self.serialize_transitions()
        state = (
            self._serialized_transitions,
            infos,
            self.iw.width,
            self.iw.expansion_strategy,
            self.problem.filepath,
            self.problem.domain.filepath,
            self.space_options,
        )
        return state

    def __setstate__(self, state):
        (
            serialized_transitions,
            state_info,
            width,
            expansion_strategy,
            problem_path,
            domain_path,
            space_options,
        ) = state
        self.__init__(
            IWSearch(width, expansion_strategy),
            domain_path,
            problem_path,
            serialized_transitions=serialized_transitions,
            serialized_state_infos=state_info,
            **space_options,
        )

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

    def serialize_transitions(self):
        """
        Serializes the IWStateSpace transition to index-based transitions.

        Pray to god that these indices never change.
        """
        for state in self:
            for transition in self.forward_transitions(state):
                action_indices = (
                    tuple(action.index for action in transition.action)
                    if isinstance(transition.action, Sequence)
                    else (transition.action.index,)
                )
                self._serialized_transitions.append(
                    (
                        state.index,
                        action_indices,
                        transition.target.index,
                    )
                )


if __name__ == "__main__":
    import os

    source_dir = Path("" if os.getcwd().endswith("/test") else "test/")
    domain = "blocks"
    # domain = "spanner"
    problem = "medium"
    domain_filepath = source_dir / "pddl_instances" / domain / "domain.pddl"
    problem_filepath = source_dir / "pddl_instances" / domain / f"{problem}.pddl"
    start_time = datetime.fromtimestamp(time.time())
    state_space = XStateSpace(domain_filepath, problem_filepath)
    iw_space = IWStateSpace(
        # IWSearch(2),
        IWSearch(2, depth_1_is_novel=False),
        state_space,
        # n_cpus=mp.cpu_count(),
        n_cpus=1,
        chunk_size=300,
    )
    elapsed = datetime.fromtimestamp(time.time()) - start_time
    hours = elapsed.total_seconds() / 3600
    minutes, seconds = divmod(elapsed.total_seconds(), 60)
    print(f"Elapsed time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
