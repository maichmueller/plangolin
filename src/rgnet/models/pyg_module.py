from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Dict, Mapping, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.typing import Adj


class PyGModule(Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor: ...

    @singledispatchmethod
    def __call__(self, *args):
        if args:
            msg = f"Invalid input type {args[0].__class__} for '__call__'"
        else:
            msg = "No input for '__call__'"
        raise NotImplementedError(msg)

    @__call__.register(dict)
    def _(self, x, edge_index, batch=None):
        return super().__call__(x, edge_index, batch)

    @__call__.register(Data)
    @__call__.register(Batch)
    def _(self, data, *args):
        return super().__call__(*self.unpack(data))

    @classmethod
    def unpack(cls, data: Union[Data, Batch]):
        return data.x, data.edge_index, data.batch


class PyGHeteroModule(Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ): ...

    @singledispatchmethod
    def __call__(self, *args):
        if args:
            msg = f"Invalid input type {args[0].__class__} for '__call__'"
        else:
            msg = "No input for '__call__'"
        raise NotImplementedError(msg)

    @__call__.register(dict)
    def _(self, x_dict, edge_index_dict, batch_dict=None):
        return super().__call__(x_dict, edge_index_dict, batch_dict)

    @__call__.register(HeteroData)
    @__call__.register(Batch)
    def _(self, data, *args):
        return super().__call__(*self.unpack(data))

    @classmethod
    def unpack(cls, data: Union[HeteroData, Batch]):
        return data.x_dict, data.edge_index_dict, data.batch_dict
