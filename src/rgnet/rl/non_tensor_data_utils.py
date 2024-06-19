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


def non_tensor_to_list(input_: Union[NonTensorData, NonTensorStack, List]):
    if isinstance(input_, (NonTensorData, NonTensorStack)):
        return input_.tolist()
    assert isinstance(input_, List)
    return input_
