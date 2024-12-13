from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.typing import Adj


class PyGModule(Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor: ...

    @classmethod
    def unpack(cls, data: Union[Data, Batch]):
        return data.x, data.edge_index, data.batch  # type: ignore

    def invoke(self, data: Union[Data, Batch]):
        return self(*self.unpack(data))


class PyGHeteroModule(Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ): ...

    @classmethod
    def unpack(cls, data: Union[HeteroData, Batch]):
        return data.x_dict, data.edge_index_dict, data.batch_dict  # type: ignore

    def invoke(self, data: Union[HeteroData, Batch]):
        return self(*self.unpack(data))
