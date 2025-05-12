from tensordict import NonTensorStack

from rgnet.utils.misc import as_non_tensor_stack


def test_as_non_tensor_stack():

    sequence = ["a", "b"]
    stack = as_non_tensor_stack(sequence)
    assert isinstance(stack, NonTensorStack)
    assert stack.batch_size == (2,)

    list_of_list = [["a"], ["b"]]
    stack = as_non_tensor_stack(list_of_list)
    assert isinstance(stack, NonTensorStack)
    assert stack.batch_size == (2,)
