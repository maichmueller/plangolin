import abc
from abc import ABCMeta
from typing import Optional

from pymimir import Problem, State


class Encoder(metaclass=ABCMeta):

    @abc.abstractmethod
    def encode(self, state: State, problem: Optional[Problem] = None):
        pass

    @abc.abstractmethod
    def encoding_to_pyg_data(self, state: State, problem: Optional[Problem] = None):
        pass
