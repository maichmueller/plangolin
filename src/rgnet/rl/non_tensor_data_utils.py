from functools import singledispatch
from typing import Iterable, List, Union

from tensordict import NonTensorData, NonTensorStack
from torch_geometric.data.batch import Batch

NonTensorWrapper = Union[NonTensorData, NonTensorStack]


def as_non_tensor_stack(sequence: Iterable) -> NonTensorStack:
    """
    Wrap every element of the list in a NonTensorData and stacks them into a
    NonTensorDataStack. We do not use torch.stack() in order to avoid getting
    NonTensorData returned, which is the case if all elements of the list are equal.
    """
    return NonTensorStack(*(NonTensorData(x) for x in sequence))


@singledispatch
def tolist(input_, **kwargs) -> List:
    return list(input_)


@tolist.register(list)
def _(input_: list, *, ensure_copy: bool = False, **kwargs) -> List:
    if ensure_copy:
        return input_.copy()
    return input_


@tolist.register(NonTensorStack)
@tolist.register(NonTensorData)
def _(input_: NonTensorWrapper, **kwargs) -> List:
    return input_.tolist()


@tolist.register(Batch)
def _(input_: Batch, **kwargs) -> List:
    return input_.to_data_list()
