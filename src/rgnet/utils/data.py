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


class CachingDataLoader(DataLoader):
    """
    A DataLoader that, on the first full iteration, collects (and stores)
    every batch yielded by the parent DataLoader; on subsequent iterations,
    yields directly from that cached list of batches.

    Note:
    - shuffle should be False if you want identical batches each epoch.
    - num_workers will only apply on the first epoch.
    """

    def __init__(self, *args, cache_batches: bool = False, **kwargs):
        """
        All args/kwargs are passed through to the usual DataLoader constructor.
        :param cache_batches: whether to cache batches on first iteration.
        """
        kwargs["persistent_workers"] = (
            False  # disable persistent workers to avoid overaccumulating memory, only output should remain persistent
        )
        super().__init__(*args, **kwargs)
        self.cache_batches = cache_batches
        # will become List[batch] after first full pass
        self._batch_cache: list | None = None

    def __iter__(self):
        # If caching is disabled, just delegate entirely
        if not self.cache_batches:
            yield from super().__iter__()
            return

        # If we haven’t built the cache yet, do so now
        if self._batch_cache is None:
            self._batch_cache = []
            for batch in super().__iter__():
                # store and yield
                self._batch_cache.append(batch)
                yield batch
        else:
            # replay the cached batches
            yield from self._batch_cache

    def __len__(self):
        # After caching, len() should reflect number of cached batches
        if self.cache_batches and self._batch_cache is not None:
            return len(self._batch_cache)
        # Otherwise, fall back to parent (i.e. number of batches estimation)
        return super().__len__()

    def clear_cache(self):
        """Discard any stored batches so that the next iteration re-builds the cache."""
        self._batch_cache = None
