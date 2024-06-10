import re

from tensordict import (
    LazyStackedTensorDict,
    NonTensorData,
    is_tensor_collection,
    unravel_key,
)
from torchrl.envs.utils import _set_single_key, _StepMDP


class _StepMDPPatch(_StepMDP):
    """Identical to _StepMDP except all calls to _step are replaced with _step_patch"""

    def __call__(self, tensordict):
        if isinstance(tensordict, LazyStackedTensorDict):
            out = LazyStackedTensorDict.lazy_stack(
                [self.__call__(td) for td in tensordict.tensordicts],
                tensordict.stack_dim,
            )
            return out

        next_td = tensordict._get_str("next", None)
        if self.validate(tensordict):
            if self.keep_other:
                out = self._exclude(self.exclude_from_root, tensordict, out=None)
            else:
                out = next_td.empty()
                self._grab_and_place(
                    self.keys_from_root,
                    tensordict,
                    out,
                    _allow_absent_keys=self._allow_absent_keys,
                )
            self._grab_and_place(
                self.keys_from_next,
                next_td,
                out,
                _allow_absent_keys=self._allow_absent_keys,
            )
            return out
        else:
            out = next_td.empty()
            total_key = ()
            if self.keep_other:
                for key in tensordict.keys():
                    if key != "next":
                        _step_patched(tensordict, out, key, total_key, self.excluded)
            elif not self.exclude_action:
                for action_key in self.action_keys:
                    _set_single_key(tensordict, out, action_key)
            for key in next_td.keys():
                _step_patched(next_td, out, key, total_key, self.excluded)
            return out


def _step_patched(source, dest, key, total_key, excluded):
    total_key = total_key + (key,)
    non_empty = False
    if unravel_key(total_key) not in excluded:
        try:
            val = source.get(key)

            if is_tensor_collection(val) and not isinstance(val, NonTensorData):
                # if val is a tensordict we need to copy the structure
                new_val = dest.get(key, None)
                if new_val is None:
                    new_val = val.empty()
                non_empty_local = False
                for subkey in val.keys():
                    non_empty_local = (
                        _step_patched(val, new_val, subkey, total_key, excluded)
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
                        _step_patched(s_td, d_td, key, total_key, excluded)
                        or non_empty_local
                    )
                non_empty = non_empty_local
            else:
                raise err

    return non_empty
