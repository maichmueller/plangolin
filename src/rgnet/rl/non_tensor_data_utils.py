from functools import singledispatch
from typing import List, Sequence, Union

from tensordict import NonTensorData, NonTensorStack

NonTensorWrapper = Union[NonTensorData, NonTensorStack]


def as_non_tensor_stack(sequence: Sequence) -> NonTensorStack:
    """
    Wrap every element of the list in a NonTensorData and stacks them into a
    NonTensorDataStack. We do not use torch.stack() in order to avoid getting
    NonTensorData returned, which is the case if all elements of the list are equal.
    """
    return NonTensorStack(*(NonTensorData(x) for x in sequence))


@singledispatch
def tolist(input_) -> List:
    return list(input_)


@tolist.register(NonTensorStack)
@tolist.register(NonTensorData)
def _(input_: NonTensorWrapper) -> List:
    return input_.tolist()
