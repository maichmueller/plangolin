import torch
from tensordict import NonTensorData, NonTensorStack
from tensordict.base import NO_DEFAULT

from rgnet.utils.misc import tolist


def patched_stack_non_tensor(list_of_non_tensor, dim=0):
    """
    Instead of checking whether all elements are equal and potentially returning
    a NonTensorData we allways return a NonTensorStack
    https://github.com/pytorch/tensordict/issues/831
    """
    return NonTensorStack(*list_of_non_tensor, stack_dim=dim)


NonTensorData._stack_non_tensor = patched_stack_non_tensor


def patched__post_init__(self):
    _tensordict = self.__dict__["_tensordict"]
    _non_tensordict = self.__dict__["_non_tensordict"]
    data = _non_tensordict.get("data", NO_DEFAULT)
    if data is NO_DEFAULT:
        data = _tensordict._get_str("data", default=NO_DEFAULT)
        if isinstance(data, torch.Tensor):  # difference to original implementation
            data_inner = data
        else:
            # If we call .data on a tensor we remove the gradients!
            data_inner = getattr(data, "data", None)
        if data_inner is None:
            # Support for stacks
            data_inner = tolist(data)
        del _tensordict["data"]
        _non_tensordict["data"] = data_inner
    assert _tensordict.is_empty(), self._tensordict


NonTensorData.__post_init__ = patched__post_init__
