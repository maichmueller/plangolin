from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Dict, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.typing import Adj

from rgnet.models.mixins import DeviceAwareMixin


class PyGModule(DeviceAwareMixin, Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor = None,
        **kwargs,
    ) -> Tensor: ...

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


class PyGHeteroModule(DeviceAwareMixin, Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        **kwargs,
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
