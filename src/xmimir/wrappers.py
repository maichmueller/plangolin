from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import chain
from typing import Any, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Union

from multimethod import multimethod
from pymimir import *


class XCategory(Enum):
    static = 0
    fluent = 1
    derived = 2


class BaseHashMixin:
    base: Any

    def __hash__(self):
        return hash(self.base)


class BaseEqMixin:
    """
    Base mixin class for equality comparison.

    Note that by providing __eq__, but no __hash__ method, python will automatically set the hash to None
    for a child class which does not define a custom __hash__ function. In order to retain the hash of
    another Mixin class, the hash mixin class needs to be inherited from FIRST to ensure correct MRO.
    Otherwise, the child class will need to provide an explicit __hash__ function itself everytime, despite
    inheriting from a mixin.
    """

    base: Any

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.base == other.base


@dataclass(slots=True, unsafe_hash=True)
class XPredicate(BaseHashMixin, BaseEqMixin):
    base: Predicate
    category: XCategory

    def __init__(self, predicate: Predicate) -> XPredicate:
        if isinstance(predicate, FluentPredicate):
            category = XCategory.fluent
        elif isinstance(predicate, DerivedPredicate):
            category = XCategory.derived
        else:
            category = XCategory.static
        self.base = predicate
        self.category = category

    @property
    def name(self):
        return self.base.get_name()

    @property
    def arity(self):
        return self.base.get_arity()

    def __str__(self):
        return f"{self.name}[{self.category.name[0].capitalize()}]/{self.arity}"


@dataclass(slots=True, unsafe_hash=True)
class XAtom(BaseHashMixin, BaseEqMixin):
    base: GroundAtom
    predicate: XPredicate

    def __init__(self, atom: GroundAtom) -> XAtom:
        self.base = atom
        self.predicate = XPredicate(atom.get_predicate())

    @property
    def objects(self) -> Sequence[Object]:
        return self.base.get_objects()

    def __str__(self):
        obj_section = " ".join(obj.get_name() for obj in self.objects)
        return f"({self.predicate.name} {obj_section})"


@dataclass(slots=True, eq=True)
class XLiteral:
    base: GroundLiteral | None
    atom: XAtom
    is_negated: bool

    def __init__(self, literal: GroundLiteral | tuple[bool, XAtom]):
        if isinstance(literal, tuple):
            self.base = None
            self.is_negated, self.atom = literal
        else:
            self.base = literal
            self.atom = XAtom(literal.get_atom())
            self.is_negated = literal.is_negated()

    def __str__(self):
        return f"{'^' if self.base.is_negated() else ''}{self.atom}"

    def __hash__(self):
        return hash((self.is_negated, self.atom))


@dataclass(eq=True, frozen=True)
class XDomain(BaseHashMixin, BaseEqMixin):
    base: Domain

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def filepath(self) -> str:
        return self.base.get_filepath()

    @cache
    def predicates(self, *category: XCategory) -> tuple[XPredicate, ...]:
        if not category:
            category = XCategory.__members__.values()

        iterable = chain.from_iterable(
            getattr(self.base, f"get_{cat.name}_predicates")() for cat in category
        )
        return tuple(XPredicate(p) for p in iterable)

    @cached_property
    def constants(self) -> ObjectList:
        return self.base.get_constants()

    @cached_property
    def actions(self) -> ActionList:
        return self.base.get_actions()

    @cached_property
    def functions(self) -> FunctionSkeletonList:
        return self.base.get_functions()

    @cached_property
    def requirements(self) -> Requirements:
        return self.base.get_requirements()


class XProblem(BaseHashMixin, BaseEqMixin):
    base: Problem
    repositories: PDDLRepositories

    @multimethod
    def __init__(self, problem: Problem, repositories: PDDLRepositories):
        self.base = problem
        self.repositories = repositories

    @multimethod
    def __init__(self, space: StateSpace):
        self.__init__(
            space.get_problem(),
            space.get_pddl_repositories(),
        )

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def filepath(self) -> str:
        return self.base.get_filepath()

    @property
    def domain(self):
        return XDomain(self.base.get_domain())

    @property
    def objects(self) -> ObjectList:
        return self.base.get_objects()

    def goal(self, *category: XCategory) -> Iterable[XLiteral]:
        if not category:
            category = XCategory.__members__.values()

        iterable = chain.from_iterable(
            getattr(self.base, f"get_{cat.name}_goal_condition")() for cat in category
        )
        return (XLiteral(l) for l in iterable)

    def initial_literals(self) -> Iterable[XLiteral]:
        """
        Get the initial literals of the problem definition.
        """
        return (
            XLiteral(literal)
            for literal in chain(
                self.base.get_static_initial_literals(),
                self.base.get_fluent_initial_literals(),
            )
        )

    def initial_atoms(self) -> Iterable[XAtom]:
        """
        Get the initial atoms of the problem definition.
        """
        return (
            XAtom(atom)
            for atom in chain(
                self.base.get_static_initial_atoms(),
                self.base.get_fluent_initial_atoms(),
            )
        )


@dataclass(eq=True)
class XState:
    base: State
    problem: XProblem

    @multimethod
    def __init__(self, state: State, problem: XProblem):
        self.base = state
        self.problem = problem

    @multimethod
    def __init__(self, index: int, space: StateSpace):
        self.__init__(space.get_vertex(index).get_state(), XProblem(space))

    @multimethod
    def __init__(self, index: int, space: XStateSpace):
        self.__init__(space[index], space.problem)

    def __iter__(self):
        return iter(self.atoms())

    def __hash__(self):
        return hash((self.base, self.problem))

    @cached_property
    def fluent_atoms(self) -> tuple[XAtom, ...]:
        """
        Get the fluent atoms of the state.

        Delayed evaluation is used to avoid unnecessary computation in case this field is never required.
        """
        return tuple(
            map(
                XAtom,
                self.problem.repositories.get_fluent_ground_atoms_from_indices(
                    self.base.get_fluent_atoms()
                ),
            )
        )

    @cached_property
    def derived_atoms(self) -> tuple[XAtom, ...]:
        """
        Get the fluent atoms of the state.

        Delayed evaluation is used to avoid unnecessary computation in case this field is never required.
        """
        return tuple(
            map(
                XAtom,
                self.problem.repositories.get_derived_ground_atoms_from_indices(
                    self.base.get_derived_atoms()
                ),
            )
        )

    @property
    def index(self) -> int:
        """
        Get the index of the state in the state space. This index is unique for each state in the state space.
        """
        return self.base.get_index()

    def atoms(self) -> Iterable[XAtom]:
        return chain(self.fluent_atoms, self.derived_atoms)

    def gather_objects(self) -> set[Object]:
        return set(chain.from_iterable(atom.objects for atom in self.atoms()))

    def satisfied_literals(self, literals: Iterable[XLiteral]) -> Iterable[XLiteral]:
        return (lit for lit in literals if self.base.literal_holds(lit.base))

    def unsatisfied_literals(self, literals: Iterable[XLiteral]) -> Iterable[XLiteral]:
        return (lit for lit in literals if not self.base.literal_holds(lit.base))

    def __str__(self):
        return (
            f"State(Index={self.index}, Fluents=["
            + ", ".join(
                map(
                    str,
                    self.fluent_atoms,
                )
            )
            + "], Derived=["
            + ", ".join(
                map(
                    str,
                    self.derived_atoms,
                )
            )
            + "])"
        )


@dataclass(slots=True, eq=True, unsafe_hash=True)
class XTransition(BaseHashMixin):
    base: Edge | None
    source: XState
    target: XState
    action: Optional[GroundAction | Iterable[GroundAction]]

    @multimethod
    def __init__(
        self,
        edge: Edge,
        space: Optional[StateSpace] = None,
    ):
        self.base = edge
        self.source = XState(edge.get_source(), space)
        self.target = XState(edge.get_target(), space)
        self.action = edge.get_creating_action()

    @multimethod
    def __init__(
        self,
        source: XState,
        target: XState,
        action: GroundAction | Iterable[GroundAction] | None,
    ):
        self.base = None
        self.source = source
        self.target = target
        self.action = action

    def __iter__(self):
        return iter((self.source, self.target, self.action))

    def __str__(self):
        return f"Transition({self.source.index} -> {self.target.index})"


class XStateSpace(BaseHashMixin, BaseEqMixin):
    """
    The extended state space class.

    Note that we implicitly rely on the underlying state space to be created with its own state repository.
    Mixing states from different repositories will lead to different states potentially holding the same index.
    This could mess up a lot of downstream operations relying on this index to be unique for a state, i.e. for a certain
    list of atoms in a problem, and not for a certain emplacement order.
    We also rely on the state space to emplace states in a fixed, deterministic order that always produces the same
    state at the same index.

    Should we want to switch this to a more flexible approach, we would need to start hashing atoms of a state to build
    a mapping of atoms, i.e. a state signature, to a state index. This would allow us to compare states from different
    repositories by their content, not by their index.
    """

    base: StateSpace
    _vertices: list[StateVertex]

    def __init__(self, space: StateSpace):
        self.base = space
        self._vertices = space.get_vertices()

    @multimethod
    def create(cls, domain_path, problem_path, **options: dict[str, str]):
        return XStateSpace(
            StateSpace.create(domain_path, problem_path, StateSpaceOptions(**options)),
        )

    @create.register
    def create(cls, problem: XProblem, **options: dict[str, str]):
        return XStateSpace.create(
            problem.domain.filepath,
            problem.filepath,
            **options,
        )

    create = classmethod(create)

    def __len__(self):
        return len(self._vertices)

    def __str__(self):
        return (
            f"StateSpace("
            f"#states={len(self)}, "
            f"#transitions={self.total_transition_count}), "
            f"#deadends={self.deadend_count}, "
            f"#goals={self.goal_count}, "
            f"solvable={self.solvable}, "
            f"solution_cost={self.goal_distance(self.initial_state())}"
        )

    def __getitem__(self, index: int) -> list[XState] | XState:
        """
        Get the state to the given index.

        Note that this method is not optimized for slicing, as it will always return a list of states.
        Also, we rely on the index to semantically mean the nth state in the repository.
        That is,
            `index == space[state].index`
        """
        if isinstance(index, slice):  # handle slicing
            return [self.get_state(i) for i in range(*index.indices(len(self)))]
        return self.get_state(index)

    def __iter__(self) -> Iterator[XState]:
        return self.states_iter()

    @property
    def problem(self) -> XProblem:
        return XProblem(self.base)

    @property
    def pddl_repositories(self) -> PDDLRepositories:
        return self.base.get_pddl_repositories()

    @property
    def solvable(self) -> bool:
        return not self.is_deadend(self.initial_state())

    @property
    def deadend_count(self) -> int:
        return self.base.get_num_deadend_vertices()

    @property
    def goal_count(self) -> int:
        return self.base.get_num_goal_vertices()

    @cached_property
    def total_transition_count(self) -> int:
        return sum(
            sum(
                1
                for _ in self.base.get_forward_adjacent_transitions(vertex.get_index())
            )
            for vertex in self._vertices
        )

    def is_deadend(self, state: XState) -> bool:
        return self.base.is_deadend_vertex(state.index)

    def states_iter(self) -> Iterator[XState]:
        return iter(XState(v.get_state(), self.problem) for v in self._vertices)

    def goal_states_iter(self) -> Iterator[XState]:
        return iter(self.get_state(idx) for idx in self.base.get_goal_vertex_indices())

    def goal_distance(self, state: XState) -> float:
        return self.base.get_goal_distance(state.index)

    def is_goal(self, state: XState) -> bool:
        return self.base.is_goal_vertex(state.index)

    def get_state(self, index: int) -> XState:
        return XState(index, self.base)

    def initial_state(self) -> XState:
        return self.get_state(self.base.get_initial_vertex_index())

    def forward_transitions(self, state: XState) -> Iterator[XTransition]:
        return self._transitions(state, "forward")

    def backward_transitions(self, state: XState) -> Iterator[XTransition]:
        return self._transitions(state, "backward")

    def _transitions(self, state: XState, direction: str) -> Iterator[XTransition]:
        return iter(
            XTransition(edge, self.base)
            for edge in getattr(self.base, f"get_{direction}_adjacent_transitions")(
                state.index
            )
        )

    def forward_transition_count(self, state: XState) -> int:
        return self._transition_count(state, "forward")

    def backward_transition_count(self, state: XState) -> int:
        return self._transition_count(state, "backward")

    def _transition_count(self, state: XState, direction: str) -> int:
        return sum(
            1
            for _ in getattr(self.base, f"get_{direction}_adjacent_transitions")(
                state.index
            )
        )

    def breadth_first_search(self, state: XState) -> tuple[int, list[XTransition]]:
        """
        Perform a breath-first search from the given state to find the shortest path to a goal state.

        Parameters
        ----------
        state: XState,
            The state to start the search from.

        Returns
        -------
        tuple[int, list[XTransition]],
            A tuple containing the length of the shortest path and the list of transitions.
        """
        self.base.get_state_repository()
        GroundedApplicableActionGenerator()
        find_solution_brfs(
            GroundedApplicableActionGenerator(),
            self.base.get_state_repository(),
            self.initial_state().base,
        )
        goals_left_to_visit = set(self.goal_states_iter())
        visited = set()
        queue = deque([(state, 0, [])])
        shortest_path = []
        shortest_len = float("inf")
        while queue:
            current, length, path = queue.pop(0)
            if self.is_goal(current):
                if not goals_left_to_visit:
                    return length
                goals_left_to_visit.remove(current)
                if length < shortest_len:
                    shortest_len = length
                    shortest_path = path
            visited.add(current)
            for transition in self.forward_transitions(current):
                if transition.target not in visited:
                    next_path = path.copy()
                    next_path.append(transition)
                    queue.append((transition.target, length + 1, next_path))
        return shortest_len, shortest_path

    def breadth_first_search_pym(self, state: XState | None = None) -> SearchResult:
        """
        Perform a breath-first search from the given state to find the shortest path to a goal state.

        Parameters
        ----------
        state: XState,
            The state to start the search from.

        Returns
        -------
        tuple[int, list[XTransition]],
            A tuple containing the length of the shortest path and the list of transitions.
        """
        if state is None:
            state = self.initial_state()
        return find_solution_brfs(
            GroundedApplicableActionGenerator(),
            self.base.get_state_repository(),
            state.base,
        )


Edge = Union[EmptyEdge, GroundActionEdge, GroundActionsEdge]

__all__ = [
    "Edge",
    "XTransition",
    "XStateSpace",
    "XProblem",
    "XDomain",
    "XState",
    "XLiteral",
    "XAtom",
    "XPredicate",
    "XCategory",
]
