from .logging_setup import setup_logger

setup_logger(__name__)

import torch
import torch_geometric as pyg

from . import encoding, models, rl, supervised

# Register custom classes for serialization with torch > 2.6.
# This is necessary for loading models that use custom classes (ours, pyg). Before 2.6 this prompted a warning, from
# 2.6 onwards it raises an error.
# https://pytorch.org/docs/2.6/notes/serialization.html#torch-load-with-weights-only-true
# https://pytorch.org/docs/2.6/generated/torch.load.html#torch-load
# This list is not exhaustive! If more features from pyg or our code are used in torch.save methods,
# they need to be added here to not trigger an exception.
torch.serialization.add_safe_globals(
    [
        supervised.MultiInstanceSupervisedSet,
        rl.data.FlashDrive,
        rl.data.AtomDrive,
        rl.data.BaseDrive,
        pyg.data.Data,
        pyg.data.HeteroData,
        pyg.data.Batch,
    ]
)

__all__ = ["encoding", "models", "rl", "supervised"]
