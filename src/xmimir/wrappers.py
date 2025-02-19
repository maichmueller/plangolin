from __future__ import annotations

from enum import Enum
from functools import cache, cached_property
from itertools import chain
from pathlib import Path
from typing import (
    Generator,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from multimethod import multimethod
from pymimir import *
from pymimir.hints import *  # because PyCharm needs some help


class XCategory(Enum):
    static = 0
    fluent = 1
    derived = 2


def hollow_check(func):
    def wrapper(self, *args, **kwargs):
        if self.is_hollow:
            raise ValueError(f"Cannot perform operation {func} on hollow object.")
        return func(self, *args, **kwargs)

    return wrapper


T = TypeVar("T")


class MimirWrapper(Generic[T]):
    """
    A mixin class to provide a base accessor that checks against hollow-ness of the underlying object.
    If not overridden de by the subclass, the equivalence check as well as the hash of the class
     are determined by the pymimir base object.
    Hollow refers to an instance without a pymimir base. This is useful for some classes
     like XTransition where you might want to manually create instances of (source, action, target).
    """

    _base: T | None

    def __init__(self, base: T | None):
        self._base = base

    @property
    def is_hollow(self):
        return self._base is None

    @property
    @hollow_check
    def base(self) -> T:
        return self._base

    @hollow_check
    def __hash__(self):
        return hash(self.base)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.base == other.base


class XPredicate(MimirWrapper[Predicate]):
    category: XCategory

    def __init__(self, predicate: Predicate):
        if isinstance(predicate, FluentPredicate):
            category = XCategory.fluent
        elif isinstance(predicate, DerivedPredicate):
            category = XCategory.derived
        else:
            category = XCategory.static
        super().__init__(predicate)
        self.category = category

    @property
    def name(self):
        return self.base.get_name()

    @property
    def arity(self):
        return self.base.get_arity()

    def __str__(self):
        return f"{self.name}[{self.category.name[0].capitalize()}]/{self.arity}"


class XAtom(MimirWrapper[GroundAtom]):
    predicate: XPredicate

    def __init__(self, atom: GroundAtom):
        super().__init__(atom)
        self.predicate = XPredicate(atom.get_predicate())

    @property
    def objects(self) -> Sequence[Object]:
        return self.base.get_objects()

    def __str__(self):
        obj_section = " ".join(obj.get_name() for obj in self.objects)
        return f"({self.predicate.name} {obj_section})"


class XLiteral(MimirWrapper[GroundLiteral]):
    atom: XAtom
    is_negated: bool

    def __init__(self, literal: GroundLiteral | tuple[bool, XAtom]):
        if isinstance(literal, tuple):
            super().__init__(None)
            self.is_negated, self.atom = literal
        else:
            super().__init__(literal)
            self.atom = XAtom(literal.get_atom())
            self.is_negated = literal.is_negated()

    def __str__(self):
        return f"{'-' if self.base.is_negated() else '+'}{self.atom}"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.is_negated == other.is_negated and self.atom == other.atom

    def __hash__(self):
        return hash((self.is_negated, self.atom))


class XDomain(MimirWrapper[Domain]):

    def __init__(self, domain: Domain):
        super().__init__(domain)

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

    def __str__(self):
        return str(self.base)


class XProblem(MimirWrapper[Problem]):
    repositories: PDDLRepositories

    @multimethod
    def __init__(self, problem: Problem, repositories: PDDLRepositories):
        super().__init__(problem)
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

    def static_atoms(self) -> Iterable[XAtom]:
        return (
            XAtom(literal.get_atom())
            for literal in self.base.get_static_initial_literals()
        )

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

    def __str__(self):
        return (
            f"Problem: {{\n"
            f"name: {self.name} ({Path(self.filepath).absolute()}),\n"
            f"domain: {self.domain.name},\n"
            f"objects: {self.objects},\n"
            f"initial: {' ∧ '.join(str(a) for a in self.initial_atoms())},\n"
            f"goal: {' ∧ '.join(str(l) for l in self.goal())}\n"
            f"}}"
        )


class XAction(MimirWrapper[GroundAction]):
    problem: XProblem

    def __init__(self, action: GroundAction, problem: XProblem):
        super().__init__(action)
        self.problem = problem

    def __str__(self):
        return self.base.to_string(self.problem.repositories)

    def str(self, for_plan=True):
        return getattr(self.base, f"to_string{'_for_plan' if for_plan else ''}")(
            self.problem.repositories
        )

    @cached_property
    def action_schema(self):
        return self.problem.domain.actions[self.base.get_action_index()]

    @property
    def name(self) -> str:
        return self.action_schema.get_name()

    @property
    def cost(self) -> float:
        return self.base.get_strips_effect().get_cost()

    def preconditions(
        self, *category: XCategory, positive: bool = True
    ) -> Generator[XAtom]:
        conditions = self.base.get_strips_precondition()
        if not category:
            category = XCategory.__members__.values()

        qualifier = "positive" if positive else "negative"

        for category in category:
            cat_name = category.name
            atom_callback = getattr(
                self.problem.repositories, f"get_{cat_name}_ground_atoms_from_indices"
            )
            condition_indices = getattr(
                conditions, f"get_{cat_name}_{qualifier}_condition"
            )
            condition_atoms = atom_callback(condition_indices)
            yield from chain(map(XAtom, condition_atoms))

    def effects(self, *category: XCategory, positive: bool = True) -> Generator[XAtom]:
        effects = self.base.get_strips_effect()
        if not category:
            category = XCategory.__members__.values()

        qualifier = "positive" if positive else "negative"

        for category in category:
            cat_name = category.name
            atom_callback = getattr(
                self.problem.repositories, f"get_{cat_name}_ground_atoms_from_indices"
            )
            effect_indices = getattr(effects, f"get_{qualifier}_effects")

            effect_atoms = atom_callback(effect_indices)
            yield from chain(map(XAtom, effect_atoms))

    @property
    def objects(self):
        objs = self.problem.objects
        return [objs[i] for i in self.base.get_object_indices()]


class XState(MimirWrapper[State]):
    problem: XProblem

    @multimethod
    def __init__(self, state: State, problem: XProblem):
        super().__init__(state)
        self.problem = problem

    @multimethod
    def __init__(self, index: int, space: StateSpace):
        self.__init__(space.get_vertex(index).get_state(), XProblem(space))

    @multimethod
    def __init__(self, index: int, space: XStateSpace):
        self.__init__(space.base.get_vertex(index).get_state(), space.problem)

    def __iter__(self):
        return iter(self.atoms())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.base == other.base and self.problem == other.problem

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

    def is_goal(self) -> bool:
        for _ in self.unsatisfied_literals(
            self.problem.goal(XCategory.fluent, XCategory.derived)
        ):
            return False
        return True

    def atoms(self, with_statics: bool = False) -> Iterable[XAtom]:
        return chain(
            (self.problem.static_atoms() if with_statics else tuple()),
            self.fluent_atoms,
            self.derived_atoms,
        )

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


class XTransition(MimirWrapper[GroundActionEdge]):
    source: XState
    target: XState
    action: Optional[XAction | tuple[XAction]]

    @multimethod
    def __init__(
        self,
        edge: GroundActionEdge | None,
        source: XState,
        target: XState,
        action: XAction | Iterable[XAction] | None,
    ):
        super().__init__(edge)
        self.source = source
        self.target = target
        self.action = tuple(action) if hasattr(action, "__iter__") else action

    @multimethod
    def __init__(
        self,
        edge: GroundActionEdge,
        space: XStateSpace,
    ):
        self.__init__(
            edge,
            XState(edge.get_source(), space.base),
            XState(edge.get_target(), space.base),
            XAction(edge.get_creating_action(), space.problem),
        )

    def __iter__(self):
        return iter((self.source, self.target, self.action))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.source == other.source
            and self.target == other.target
            and self.action == other.action
        )

    def __hash__(self):
        return hash((self.source, self.target, self.action))

    def __str__(self):
        return f"Transition({self.source.index} -> {self.target.index})"

    def explain(self) -> str:
        fluents_before = set(self.source.fluent_atoms)
        fluents_after = set(self.target.fluent_atoms)
        derived_before = set(self.source.derived_atoms)
        derived_after = set(self.target.derived_atoms)
        added_fluents = fluents_after - fluents_before
        removed_fluents = fluents_before - fluents_after
        added_derived = derived_after - derived_before
        removed_derived = derived_before - derived_after
        return (
            f"Transition({self.source.index} -> {self.target.index})\n"
            f"Action: {self.action}\n"
            f"Added Fluent Atoms:    {added_fluents}\n"
            f"Added Derived Atoms:   {added_derived}\n"
            f"Deleted Fluents Atoms: {removed_fluents}\n"
            f"Removed Derived Atoms: {removed_derived}"
        )

    @cached_property
    def is_improper(self):
        return self.action is None

    @property
    def is_primitive(self):
        return isinstance(self.action, XAction)


class XActionGenerator(MimirWrapper[LiftedApplicableActionGenerator]):
    grounder: Grounder
    problem: XProblem

    @multimethod
    def __init__(self, grounder: Grounder):
        super().__init__(
            LiftedApplicableActionGenerator(grounder.get_action_grounder())
        )
        self.grounder = grounder

    @multimethod
    def __init__(self, problem: XProblem):
        self.__init__(Grounder(problem.base, problem.repositories))

    @property
    def problem(self):
        return XProblem(
            self.grounder.get_problem(), self.grounder.get_pddl_repositories()
        )

    def generate_actions(self, state: XState) -> Iterator[XAction]:
        for action in self.base.generate_applicable_actions(state.base):
            yield XAction(action, self.problem)

    def __hash__(self):
        return hash((self.grounder, self.problem))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.grounder == other.grounder and self.problem == other.problem


class XSuccessorGenerator(MimirWrapper[StateRepository]):
    action_generator: XActionGenerator
    grounder: Grounder

    def __init__(
        self,
        grounder: Grounder | XProblem,
        state_repository: StateRepository | None = None,
        action_generator: XActionGenerator | None = None,
    ):
        if isinstance(grounder, XProblem):
            grounder = Grounder(grounder.base, grounder.repositories)
        self.grounder = grounder
        self.action_generator = action_generator or XActionGenerator(grounder)
        state_repository = state_repository or StateRepository(
            LiftedAxiomEvaluator(grounder.get_axiom_grounder())
        )
        super().__init__(state_repository)

    @property
    def problem(self):
        return XProblem(
            self.grounder.get_problem(), self.grounder.get_pddl_repositories()
        )

    @property
    def initial_state(self) -> XState:
        return XState(
            self.base.get_or_create_initial_state(),
            self.problem,
        )

    def successor(self, state: XState, action: XAction) -> XState:
        return XState(
            self.base.get_or_create_successor_state(
                state.base,
                action.base,
            )[0],
            state.problem,
        )

    def successors(self, state: XState) -> Generator[tuple[XAction, XState]]:
        for action in self.action_generator.generate_actions(state):
            yield action, self.successor(state, action)

    __hash__ = None


class XSearchResult(MimirWrapper[SearchResult]):
    start: XState
    problem: XProblem

    @property
    def status(self):
        return self.base.status

    @cached_property
    def plan(self) -> tuple[XAction, ...] | None:
        if self.base.plan is not None:
            plan = self.base.plan
            return tuple(XAction(action, self.problem) for action in plan.get_actions())
        return None

    @property
    def cost(self) -> float | None:
        if self.base.plan is not None:
            return self.base.plan.get_cost()
        return None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.base == other.base
            and self.problem == other.problem
            and self.start == other.start
        )

    __hash__ = None


class XStateSpace(MimirWrapper[StateSpace]):
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

    _vertices: list[StateVertex]

    def __init__(self, space: StateSpace):
        super().__init__(space)
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
            f"solution_cost={self.goal_distance(self.initial_state)}"
        )

    def str(self):
        return str(self.base)

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
        return not self.is_deadend(self.initial_state)

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

    @cached_property
    def initial_state(self) -> XState:
        return self.get_state(self.base.get_initial_vertex_index())

    def forward_transitions(self, state: XState) -> Iterator[XTransition]:
        return self._transitions(state, "forward")

    def backward_transitions(self, state: XState) -> Iterator[XTransition]:
        return self._transitions(state, "backward")

    def _transitions(self, state: XState, direction: str) -> Iterator[XTransition]:
        return iter(
            XTransition(edge, self)
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

    def breadth_first_search(self, state: XState | None = None) -> XSearchResult:
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
            state = self.initial_state

        return XSearchResult(
            find_solution_brfs(
                XActionGenerator(self.problem).base,
                self.base.get_state_repository(),
                state.base,
            )
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
    "XAction",
    "XActionGenerator",
    "XSuccessorGenerator",
]
