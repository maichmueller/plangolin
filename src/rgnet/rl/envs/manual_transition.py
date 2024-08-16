import abc
from typing import Optional

import pymimir as mi


class _TransitionMock(abc.ABCMeta):
    def __instancecheck__(self, instance):
        return isinstance(instance, mi.Transition) or isinstance(instance, MTransition)


class MTransition(metaclass=_TransitionMock):
    """There is sadly no constructor for mi.Transition, which is really just a data class.
    As there is no constructor we can't inherit from it, so we have to mock it."""

    def __init__(self, source: mi.State, action: Optional[mi.Action], target: mi.State):
        super().__init__()
        self._source = source
        self._action = action
        self._target = target

    @property
    def action(self):
        return self._action

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __eq__(self, __value):
        return isinstance(__value, mi.Transition) and (
            __value.action == self.action
            and __value.source == self.source
            and __value.target == self.target
        )

    def __hash__(self):
        return hash((self.source, self.action, self.target))
