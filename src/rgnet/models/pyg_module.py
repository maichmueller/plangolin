from abc import ABC
from typing import Dict, Optional

from torch import Tensor
from torch.nn import Module
from torch_geometric.typing import Adj


class PyGModule(Module, ABC):
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        raise NotImplementedError(f"'forward' not implemented by {self.__class__}.")


class PyGHeteroModule(Module, ABC):
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):
        raise NotImplementedError(f"'forward' not implemented by {self.__class__}.")
