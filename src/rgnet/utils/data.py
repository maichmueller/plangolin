import queue
import threading
from abc import ABC
from queue import Queue
from typing import Any, Iterable, Union

import torch
from lightning_fabric.utilities.apply_func import (
    _BLOCKING_DEVICE_TYPES,
    _TransferableDataType,
)
from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader


class DetachableDataType(ABC):
    """A custom type for data that can be detached via ``.detach()``.

    Example:

        >>> isinstance(dict, DetachableDataType)
        False
        >>> isinstance(torch.rand(2, 3), DetachableDataType)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def detach(self):
        ...         self.x = self.x.detach()
        ...         return self
        >>> isinstance(CustomObject(), DetachableDataType)
        True

    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is _TransferableDataType:
            detach = getattr(subclass, "detach", None)
            return callable(detach)
        return NotImplemented


def transfer_batch_to_device(batch, device, *_, **__):
    """
    Transfers a batch of data to the specified device.

    This is a PyG data friendly version of lightning's function,
    since it does not restrict the data object to be of type `tensor`.
    """
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:
        kwargs = {}
        # Don't issue non-blocking transfers to CPU
        # Same with MPS due to a race condition bug: https://github.com/pytorch/pytorch/issues/83015
        if device.type not in _BLOCKING_DEVICE_TYPES:
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        return data

    return apply_to_collection(batch, dtype=_TransferableDataType, function=batch_to)


def recursively_detach(batch: Any) -> Any:
    """
    Recursively detaches tensors in a batch, allowing for safe transfer to CPU or other devices.
    """
    return apply_to_collection(
        batch, dtype=DetachableDataType, function=lambda x: x.detach()
    )


class MultiSourceValDataLoader:
    """
    Wraps multiple DataLoaders into one.  Under the hood, spins up
    one thread per loader and yields (batch, batch_idx, loader_idx) as soon
    as any loader produces one.
    """

    def __init__(self, loaders: Iterable[DataLoader]):
        self.loaders = list(loaders)
        if not self.loaders:
            raise ValueError("Need at least one DataLoader")
        self.max_queue_size = sum(loader.prefetch_factor for loader in self.loaders)
        self.timeout = max(loader.timeout for loader in self.loaders)

    def __len__(self):
        # mimic DataLoader.__len__: all sub-loaders to completion
        return sum(len(dl) for dl in self.loaders)

    def __iter__(self):
        q: Queue[tuple[Any, int, int]] = Queue(self.max_queue_size)
        stop_flags = torch.zeros((len(self.loaders),), dtype=torch.bool)

        def _worker(loader_idx: int, loader: DataLoader):
            for batch_idx, batch in enumerate(loader):
                q.put((batch, batch_idx, loader_idx))
            stop_flags[loader_idx] = True

        # start one thread per loader
        threads = [
            threading.Thread(target=_worker, args=(i, dl), daemon=True)
            for i, dl in enumerate(self.loaders)
        ]
        for t in threads:
            t.start()

        # produce batches, but ensure threads get joined on exit
        try:
            while not stop_flags.all() or not q.empty():
                try:
                    yield q.get(timeout=self.timeout)
                except queue.Empty:
                    continue
                except Exception as e:
                    raise e
        finally:
            # always join threads, no matter how we exit
            for t in threads:
                t.join()

    def __getattr__(self, name):
        """
        Delegate attribute lookups to the first DataLoader.
        This covers dataset, batch_size, num_workers, pin_memory, etc.
        """
        # avoid infinite recursion
        if name in self.__dict__:
            return self.__dict__[name]
        # delegate to the first loader
        return getattr(self.loaders[0], name)


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10, MNIST, FakeData
    from torchvision.transforms import transforms

    # common transform
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    # 1) MNIST
    mnist_val = MNIST(
        root="data/mnist", train=False, download=True, transform=transform
    )
    dl_mnist = DataLoader(mnist_val, batch_size=64, num_workers=4, pin_memory=True)

    # 2) CIFAR10
    cifar_val = CIFAR10(
        root="data/cifar10", train=False, download=True, transform=transform
    )
    dl_cifar = DataLoader(cifar_val, batch_size=64, num_workers=4, pin_memory=True)

    # 4) FakeData (random noise, just for demo)
    fake_val = FakeData(
        size=500, image_size=(3, 32, 32), num_classes=10, transform=transform
    )
    dl_fake = DataLoader(fake_val, batch_size=50, num_workers=2, pin_memory=True)

    # Wrap them all
    multi_val_loader = MultiSourceValDataLoader(
        loaders=[dl_mnist, dl_cifar, dl_fake],
    )

    # Example loop
    for batch, batch_idx, loader_idx in multi_val_loader:
        imgs, labels = batch  # assuming every loader returns (data, target)
        print(
            f"From loader #{loader_idx}, batch {batch_idx}: imgs={imgs.shape}, labels={labels.shape}"
        )
        # ... do your validation forward, metric logging, etc. ...
