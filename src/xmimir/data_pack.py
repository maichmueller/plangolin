from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar, Union

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

    def __str__(self):
        """
        String representation of the action data pack.
        """
        return f"ActionDataPack({self.schema}, {self.objects})"

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

    actions: list[ActionDataPack]

    def __init__(self, action_history: Iterable[XAction]):
        self.actions = [ActionDataPack(action) for action in action_history]

    def __len__(self) -> int:
        """
        Get the number of actions in the action history.
        """
        return len(self.actions)

    def __getitem__(
        self, index: int | slice
    ) -> Union["ActionHistoryDataPack", ActionDataPack]:
        """
        Get the action data pack at the specified index.
        """
        if isinstance(index, slice):
            copied = ActionHistoryDataPack(tuple())
            copied.actions = self.actions[index]
            return copied
        return self.actions[index]

    def __bool__(self):
        """
        Check if the action history is not empty.
        """
        return bool(self.actions)

    def __str__(self):
        """
        String representation of the action history.
        """
        return (
            f"ActionHistoryDataPack({len(self.actions)}: {list(map(str,self.actions))})"
        )

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

    def extend(self, other: "ActionHistoryDataPack") -> None:
        """
        Extend the current action history with another action history.
        """
        self.actions.extend(other.actions)
