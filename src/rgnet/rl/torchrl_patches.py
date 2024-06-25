import re

import torchrl.envs.utils
from tensordict import NonTensorData, NonTensorStack, is_tensor_collection, unravel_key


def patched_stack_non_tensor(list_of_non_tensor, dim=0):
    """
    Instead of checking whether all elements are equal and potentially returning
    a NonTensorData we allways return a NonTensorStack
    https://github.com/pytorch/tensordict/issues/831
    """
    return NonTensorStack(*list_of_non_tensor, stack_dim=dim)


def _set_patched(source, dest, key, total_key, excluded):
    """
    This method is used by step_mdp and the cached version. The only difference to the
    original is the inclusion of NonTensorData and NonTensorStack.
    This should be fixed after 0.4.0.
    https://github.com/pytorch/rl/pull/1944
    """
    total_key = total_key + (key,)
    non_empty = False
    if unravel_key(total_key) not in excluded:
        try:
            val = source.get(key)

            if (
                is_tensor_collection(val)
                and not isinstance(val, NonTensorData)
                and not isinstance(val, NonTensorStack)
            ):
                # if val is a tensordict we need to copy the structure
                new_val = dest.get(key, None)
                if new_val is None:
                    new_val = val.empty()
                non_empty_local = False
                for subkey in val.keys():
                    non_empty_local = (
                        _set_patched(val, new_val, subkey, total_key, excluded)
                        or non_empty_local
                    )
                if non_empty_local:
                    # dest.set(key, new_val)
                    dest._set_str(
                        key, new_val, inplace=False, validated=True, non_blocking=False
                    )
                non_empty = non_empty_local
            else:
                non_empty = True
                # dest.set(key, val)
                dest._set_str(
                    key, val, inplace=False, validated=True, non_blocking=False
                )
        # This is a temporary solution to understand if a key is heterogeneous
        # while not having performance impact when the exception is not raised
        except RuntimeError as err:
            if re.match(r"Found more than one unique shape in the tensors", str(err)):
                # this is a het key
                non_empty_local = False
                for s_td, d_td in zip(source.tensordicts, dest.tensordicts):
                    non_empty_local = (
                        _set_patched(s_td, d_td, key, total_key, excluded)
                        or non_empty_local
                    )
                non_empty = non_empty_local
            else:
                raise err

    return non_empty


NonTensorData._stack_non_tensor = patched_stack_non_tensor

torchrl.envs.utils._set = _set_patched
