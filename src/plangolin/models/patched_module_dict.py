from typing import Dict, Iterator, List, Mapping, Tuple

import torch


class PatchedModuleDict(torch.nn.Module):
    r"""
    Torch ModuleDict wrapper that permits keys with any name.

    Torch's ModuleDict doesn't allow certain keys to be used if they
    conflict with existing class attributes, e.g.

    > torch.nn.ModuleDict({'type': torch.nn.Module()}) # Raises KeyError.
    > torch.nn.ModuleDict({'clear': torch.nn.Module()}) # Raises KeyError.
    > torch.nn.ModuleDict({'to': torch.nn.Module()}) # Raises KeyError.

    This class is a simple wrapper around torch's ModuleDict that
    mitigates possible conflicts by using a key-suffixing protocol.

    For more details, see the following issue:
    https://github.com/pytorch/pytorch/issues/71203
    """

    SUFFIX = "____"
    SUFFIX_LENGTH = len(SUFFIX)

    def __init__(self, module_map: Mapping[str, torch.nn.Module] = None) -> None:
        super().__init__()
        self.module_dict = torch.nn.ModuleDict()
        if module_map is not None:
            if not isinstance(module_map, Mapping):
                raise TypeError("dict argument must be a mapping")
            for key, module in module_map.items():
                if not isinstance(module, torch.nn.Module):
                    raise TypeError(
                        f"Value for key {key} must be a torch.nn.Module, got {type(module)}"
                    )
                self[key] = module

    def __getitem__(self, key) -> torch.nn.Module:
        return self.module_dict[key + self.SUFFIX]

    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        self.module_dict[key + self.SUFFIX] = module

    def __len__(self) -> int:
        return len(self.module_dict)

    def keys(self) -> List[str]:
        return [key[: -self.SUFFIX_LENGTH] for key in self.module_dict.keys()]

    def values(self) -> List[torch.nn.Module]:
        return [module for _, module in self.module_dict.items()]

    def items(self) -> List[Tuple[str, torch.nn.Module]]:
        return [
            (key[: -self.SUFFIX_LENGTH], module)
            for key, module in self.module_dict.items()
        ]

    def update(self, modules: Dict[str, torch.nn.Module]) -> None:
        for key, module in modules.items():
            self[key] = module

    def __next__(self) -> str:
        return next(iter(self))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())
