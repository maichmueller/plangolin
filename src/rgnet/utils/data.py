import queue
import threading
from abc import ABC
from queue import Queue
from typing import Any, Callable, Iterable, Sequence, Sized, Union

import numpy as np
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


def split(seq: Sequence[Any], spec: Sequence[int]) -> list[list[Any]]:
    """
    Split `seq` either
      - by counts if sum(spec) == len(seq), or
      - by split-indices otherwise.

    If treating `spec` as counts:
      sum(counts) must equal len(seq), and all counts >= 0.

    If treating `spec` as indices:
      indices must be sorted ascending, each 0 <= idx <= len(seq).

    Examples:
        split([1,2,3,4,5,6], [2,3,1])
         -> [[1,2], [3,4,5], [6]]

        split([1,2,3,4,5], [1,3])
         -> [[1], [2,3], [4,5]]
    """
    n = len(seq)
    total = sum(spec)

    # If spec sums to full length, treat as counts
    if total == n:
        # counts mode
        start = 0
        chunks: list[list] = []
        for c in spec:
            if c < 0:
                raise ValueError(f"counts must be non-negative, got {c}")
            end = start + c
            chunks.append(list(seq[start:end]))
            start = end
        return chunks

    # Otherwise treat as indices
    # (must be sorted and in [0, n])
    if any(i < 0 or i > n for i in spec):
        raise ValueError(f"indices must be in [0, {n}], got {spec}")
    if any(spec[i] > spec[i + 1] for i in range(len(spec) - 1)):
        raise ValueError(f"indices must be sorted ascending, got {spec}")

    chunks = []
    prev = 0
    for idx in spec:
        chunks.append(list(seq[prev:idx]))
        prev = idx
    chunks.append(list(seq[prev:]))
    return chunks


def _max_depth(seq):
    if not isinstance(seq, Sequence) or not seq:
        return 0
    return 1 + max(_max_depth(s) for s in seq)


def map_last_dim(fn: Callable[[Sequence], Any], nested: Sequence) -> list:
    """Recursively walks `nested` and applies `fn` to each innermost sequence (last dimension).

    An innermost list is defined as a list whose elements are not themselves lists. The function
    returns a new nested structure with the same shape, where each such innermost list is replaced
    with `fn(innermost_list)`.
    """
    if not isinstance(nested, Sequence):  # Not a list, nothing to map over.
        raise TypeError(f"{type(nested)} is not a sequence")
    # If this is a leaf: its elements are not lists (or it's empty), apply fn directly.
    if not nested or not any(isinstance(el, Sequence) for el in nested):
        return fn(nested)
    # Otherwise, recurse deeper.
    return [map_last_dim(fn, sub) for sub in nested]


def map_dim(
    fn: Callable[[Sequence], Any], nested: Sequence, dim: int, depth: int | None = None
) -> list:
    """Apply `fn` over the specified dimension `dim` in `nested`.

    Dimension 0 is the outermost list. If `dim` is negative, it is interpreted relative to the maximum depth
    (like Python indexing) by first computing the maximum depth of the nested sequence. The function returns a new
    nested structure with the same shape, except that at the specified dimension level each sub-sequence is replaced
    with `fn(sub_sequence)`.
    """
    if not isinstance(nested, Sequence):
        raise TypeError(f"Expected sequence for `nested`, got {type(nested)}")

    depth = _max_depth(nested) if depth is None else depth
    if dim < 0:
        dim = depth + dim + 1
    if dim < 0:
        raise ValueError(
            f"Dimension {dim} out of range for nested structure with depth {depth}"
        )

    def _recurse(curr, level):
        if level == dim:
            # Apply fn to this subsequence
            return fn(curr)
        if not isinstance(curr, Sequence):
            return curr
        return [_recurse(sub, level + 1) for sub in curr]

    return _recurse(nested, 0)


def map_every_dim(
    fn: Callable[[Sequence], Any], nested: Sequence, depth: int | None = None
) -> list:
    """Apply `fn` over every valid dimension in `nested`.

    Returns a list where the element at index i is equivalent to
    `map_dim(nested, fn, dim=i)`.
    """

    depth = _max_depth(nested) if depth is None else depth
    return [map_dim(fn, nested, dim=i, depth=depth) for i in range(depth)]


def map_nested(
    fn,
    nested,
    *,
    exclude_dims: Sequence[int] = (),
    preserve_types: bool = True,
    descend_pred=None,
    depth: int | None = None,
):
    """
    Recursively apply `fn` to leaves of a nested sequence, with optional exclusion of certain levels.

    A "leaf" is any node where either:
      * its depth is in `exclude_dims` (i.e., stop descending at that level and apply `fn` to the entire subtree),
      * or `descend_pred(node)` is False (by default, non-sequence or str/bytes),
      * otherwise, recursion continues.

    Args:
        nested: Arbitrarily nested sequence (e.g., lists/tuples) of values.
        fn: Callable applied to each leaf value or excluded subtree.
        exclude_dims: Sequence of integer depths at which to stop descent and treat the
                      entire subtree as a leaf. Negative indices are interpreted relative
                      to the full depth (like Python indexing). Root is depth 0.
        preserve_types: If True, attempts to reconstruct sequence containers with the same
                        type; falls back to `list` on failure.
        descend_pred: Optional predicate deciding whether to descend into a value. Defaults to
                      "is a Sequence but not str/bytes".
        depth: Optional precomputed maximum depth of the nested structure; if omitted it will be inferred.

    Returns:
        New nested structure with `fn` applied according to the rules.

    Examples:
        >>> data = [[1, 2], [3, 4]]
        >>> # increment scalars
        >>> map_nested(lambda x: x + 1,data)
        [[2, 3], [4, 5]]

        >>> # apply fn to inner lists as wholes by excluding depth=1
        >>> map_nested(lambda x: tuple(x),data,exclude_dims=(1,))
        [(1, 2), (3, 4)]
    """
    if descend_pred is None:
        descend_pred = lambda x: isinstance(x, Sequence) and not isinstance(
            x, (str, bytes)
        )

    # determine full depth if not provided
    if depth is None:
        depth = _max_depth(nested, descend_pred)

    # normalize exclude_dims once
    normalized = set()
    for d in exclude_dims:
        if d < 0:
            d = depth + d
        if d < 0 or d >= depth:
            raise ValueError(
                f"Excluded dimension {d} out of range for nested structure with depth {depth}"
            )
        normalized.add(d)

    def _recurse(node, current_depth):
        # If current depth is excluded, treat entire subtree as leaf
        if current_depth in normalized:
            return fn(node)
        # If we shouldn't descend here, apply fn
        if not descend_pred(node):
            return fn(node)
        # Otherwise, descend
        mapped_children = []
        for child in node:
            mapped_children.append(_recurse(child, current_depth + 1))
        if preserve_types:
            try:
                return type(node)(mapped_children)
            except KeyboardInterrupt:
                raise
            except Exception:
                return list(mapped_children)
        else:
            return list(mapped_children)

    return _recurse(nested, 0)


def map_except(
    fn: Callable,
    nested: Sequence,
    exclude_dims: Sequence[int],
    depth: int | None = None,
) -> list:
    """Apply `fn` over every dimension in `nested` except those in `exclude_dims`.

    Returns a list where for each dimension i not in `exclude_dims` the i-th entry is
    equivalent to `map_dim(nested, fn, dim=i)`.  Dimensions in `exclude_dims` are skipped.

    Negative indices in `exclude_dims` are interpreted relative to the structure's depth.
    """

    depth = _max_depth(nested) if depth is None else depth
    # Normalize exclude dims to positive indices
    normalized = set()
    for d in exclude_dims:
        if d < 0:
            d = depth + d
        if d < 0 or d >= depth:
            raise ValueError(
                f"Excluded dimension {d} out of range for nested structure with depth {depth}"
            )
        normalized.add(d)

    return [map_dim(fn, nested, dim=i) for i in range(depth) if i not in normalized]


def cascade(
    fn: Callable,
    nested: Sequence,
    exclude_dims: Sequence[int] = (),
    depth: int | None = None,
) -> list:
    """
    Cascades the function on all dimension but those in `exclude_dims`.

    Applies `fn` over the remaining dimensions in reverse order (deepest first), cascading changes,
    and returns the resulting nested structure.
    """
    depth = _max_depth(nested) if depth is None else depth
    # normalize exclude_dims
    normalized = set()
    for d in exclude_dims:
        if d < 0:
            d = depth + d + 1
        if d < 0 or d > depth:
            raise ValueError(
                f"Excluded dimension {d} out of range for nested structure with depth {depth}"
            )
        normalized.add(d)

    current = nested
    # iterate dims from deepest to outermost
    for dim in reversed(range(depth)):
        if dim in normalized:
            continue
        current = map_dim(fn, current, dim=dim)
    return current


def flat_to_array(
    outer_shape: Sequence[int], leaves: Iterable, dtype: np.dtype = None
) -> np.ndarray:
    """
    outer_shape: sequence of ints, e.g., (2,3)
    leaves: iterable of leaf values; each leaf can be a sequence (will be
            converted to tuple) or scalar. Must supply exactly prod(outer_shape)
            items.
    dtype: optional dtype for the resulting array, defaults to inferring (could break).
    Filling order is lexicographic (like np.ndindex over outer_shape).
    """
    total = int(np.prod(outer_shape))
    leaves_iter = iter(leaves)
    arr = np.empty(outer_shape, dtype=dtype)
    for idx in np.ndindex(*outer_shape):
        try:
            leaf = next(leaves_iter)
        except StopIteration:
            raise ValueError(f"Not enough leaves: expected {total}, got fewer")
        if isinstance(leaf, Sequence) and not isinstance(leaf, (str, bytes)):
            arr[idx] = tuple(leaf)
        else:
            arr[idx] = leaf
    # ensure no extra
    try:
        next(leaves_iter)
        raise ValueError(f"Too many leaves: expected {total}, got more")
    except StopIteration:
        pass
    return arr


def nested_to_array(
    outer_shape: Sequence[int], nested: Sized, dtype: np.dtype = None
) -> np.ndarray:
    """
    Construct an object array of shape `outer_shape` from a nested sequence of leaves that already
    has the same outer nesting structure. Each leaf can be a sequence (will be converted to a tuple)
    or a scalar. The nesting of `nested` must exactly match `outer_shape`.

    Args:
        outer_shape: Tuple of ints representing the desired outer shape, e.g., (2, 3).
        nested: Nested sequence of leaves with the same shape as `outer_shape`. At the deepest
            level each value is either a scalar or a sequence (converted to tuple).
        dtype: Optional dtype for the resulting array. If not provided, defaults to inferring.

    Returns:
        np.ndarray: An object-dtype array of shape `outer_shape` with each leaf
                    holding the element at that index in `nested`

    Raises:
        ValueError: If the structure of `nested` does not match `outer_shape`.
    """

    def _fill(array, node, idx_prefix=()):
        level = len(idx_prefix)
        if level == len(outer_shape):  # leaf level
            array[idx_prefix] = node
            return
        # interior level: must be sequence of correct length
        expected_len = outer_shape[level]
        if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
            raise ValueError(
                f"Expected sequence at depth {level} to match outer_shape {outer_shape}, got {type(node)}"
            )
        if len(node) != expected_len:
            raise ValueError(
                f"Length mismatch at depth {level}: expected {expected_len}, got {len(node)}"
            )
        for i, child in enumerate(node):
            _fill(array, child, idx_prefix + (i,))

    arr = np.empty(outer_shape, dtype=dtype)
    _fill(arr, nested, ())
    return arr
