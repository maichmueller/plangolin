from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

from .wrappers import *

T = TypeVar("T")


@dataclass
class MinDataPack(Generic[T], ABC):
    """
    Minimal interface for a data pack that can be used to reconstruct the original object.
    """

    @abstractmethod
    def reconstruct(self, *args, **kwargs) -> T:
        """
        Fully reconstruct the original object T from the data pack. May require additional parameters.
        """
        ...


@dataclass
class ActionDataPack(MinDataPack[XAction]):
    """
    Minimal interface for an action data pack that can be used to reconstruct the original action.
    """

    schema: str
    objects: tuple[str, ...]

    def __init__(self, action: XAction):
        self.schema = action.action_schema.name
        self.objects = tuple(map(lambda o: o.get_name(), action.objects))

    def reconstruct(self, action_generator: XActionGenerator) -> XAction:
        return action_generator.ground_action(self.schema, self.objects)


class AtomDataPack(MinDataPack[XAtom]):
    """
    Minimal interface for a ground atom data pack that can be used to reconstruct the original ground atom.
    """

    predicate: str
    objects: tuple[str, ...]

    def __init__(self, atom: XAtom):
        self.schema = atom.predicate.name
        self.objects = tuple(o.get_name() for o in atom.objects)

    def reconstruct(self, problem: XProblem) -> XAtom:
        raise NotImplementedError(
            "Reconstructing ground atoms is not possible with the current version of pymimir."
        )


@dataclass
class StateDataPack(MinDataPack[XState]):
    """
    Minimal interface for a state data pack that can be used to reconstruct the original state.
    """

    fluent_atoms: tuple[AtomDataPack, ...]

    def __init__(self, state: XState):
        self.fluent_atoms = tuple(AtomDataPack(atom) for atom in state.fluent_atoms)

    def reconstruct(self, *args, **kwargs) -> T:
        raise NotImplementedError(
            "Reconstructing states from atoms is not possible with the current version of pymimir. "
            "Use ActionHistoryDataPack to do so instead."
        )


@dataclass
class ActionHistoryDataPack(MinDataPack[XState]):
    """
    Minimal interface for an action history data pack that can be used to reconstruct the original action history.
    """

    actions: tuple[ActionDataPack, ...]

    def __init__(self, action_history: Iterable[XAction]):
        self.actions = tuple(ActionDataPack(action) for action in action_history)

    def reconstruct(self, successor_generator: XSuccessorGenerator) -> XState:
        return self.reconstruct_sequence(successor_generator)[-1]

    def reconstruct_sequence(
        self, successor_generator: XSuccessorGenerator
    ) -> list[XState]:
        """
        Reconstruct the sequence of states from the action history.
        """
        state = successor_generator.initial_state
        action_gen = successor_generator.action_generator
        states = [state]
        for action_pack in self.actions:
            action = action_pack.reconstruct(action_gen)
            state = successor_generator.successor(state, action)
            states.append(state)
        return states
