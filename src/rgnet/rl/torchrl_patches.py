from tensordict import NonTensorData, NonTensorStack


def patched_stack_non_tensor(list_of_non_tensor, dim=0):
    """
    Instead of checking whether all elements are equal and potentially returning
    a NonTensorData we allways return a NonTensorStack
    https://github.com/pytorch/tensordict/issues/831
    """
    return NonTensorStack(*list_of_non_tensor, stack_dim=dim)


NonTensorData._stack_non_tensor = patched_stack_non_tensor
