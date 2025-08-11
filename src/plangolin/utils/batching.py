import itertools
from dataclasses import dataclass
from typing import Callable, Generator, Generic, Sequence, TypeVar

import torch
from torch.utils.data import BatchSampler, ConcatDataset, Dataset, SequentialSampler

try:
    from itertools import batched
except ImportError:
    from collections.abc import Iterable, Iterator
    from itertools import islice
    from typing import List, TypeVar

    U = TypeVar("U")

    def batched(iterable: Iterable[U], n: int) -> Iterator[List[U]]:
        """Batch data into lists of length n. The last batch may be shorter.

        Equivalent enough to itertools.batched, available in Python 3.12+.
        """
        if n <= 0:
            raise ValueError("n must be > 0")
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch


T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class PermutedData(Generic[T]):
    permutation: tuple[int, ...]
    data: T | torch.Tensor

    def __iter__(self):
        yield self.permutation
        yield self.data


@dataclass(slots=True, frozen=True)
class PermutedDataBatch(Generic[T]):
    permutations: list[tuple[int, ...]]
    data: list[T | torch.Tensor]
    arity: int

    def __iter__(self):
        for perm, data in zip(self.permutations, self.data):
            yield PermutedData(perm, data)

    def __len__(self):
        return len(self.permutations)


def batched_permutations(
    x: torch.Tensor | Sequence[T],
    arity: int,
    batch_size: int | None = 1,
    transform: Callable[[list[T] | torch.Tensor], list[T] | torch.Tensor] = lambda x: x,
    with_replacement: bool = False,
) -> Generator[PermutedDataBatch[T], None, None]:
    """
    Generate batches of index-ordered permutations of the first dimension of `x`.

    For a tensor input, rows (dim=0) are permuted. For a sequence input, elements are
    selected by index. Order matters; without replacement this is equivalent to
    `itertools.permutations(range(N), r=arity)`, with replacement to
    `itertools.product(range(N), repeat=arity)`.

    Args:
        x:
            Either
            - a tensor with shape ``[N, ...]`` where the first dimension is permuted, or
            - a sequence of length ``N`` whose elements are indexed.
        arity:
            Length of each permutation tuple ``p = (i0, ..., i_{arity-1})``.
        batch_size:
            Number of permutations per yielded batch. If ``None``, stream all permutations
            in a single batch.
        transform:
            Function applied to each permuted slice before yielding. It receives:
            - for tensor input: ``x[p]`` with shape ``[arity, ...]``
            - for sequence input: ``[x[i] for i in p]``
            and must return an object of the same 'container kind' (tensor or list).
        with_replacement:
            If ``True``, indices may repeat within a permutation.

    Yields:
        PermutedDataBatch:
            An object with fields:
            - ``permutations: list[tuple[int, ...]]`` of length ``<= batch_size``
            - ``data: list[Tensor | T]`` transformed per permutation
            - ``arity: int`` equal to the provided ``arity``

    Notes:
        - If ``arity > N`` and ``with_replacement=False``, the generator yields nothing.
        - Total number of permutations is ``P(N, arity) = N!/(N-arity)!`` without replacement,
          and ``N**arity`` with replacement. Use ``batch_size`` to bound memory.
    """

    def yield_batch(batch_: list[PermutedData[T]]) -> PermutedDataBatch[T]:
        perms, data = zip(*batch_)
        return PermutedDataBatch(perms, data, arity=arity)

    def sliceit(data: torch.Tensor | T, indices: Sequence[int]) -> T | torch.Tensor:
        if isinstance(x, torch.Tensor):
            return data[list(indices)]
        else:
            return [data[i] for i in indices]

    if isinstance(x, torch.Tensor):
        assert x.ndim >= 2, f"Input must be at least 2D tensor. Given {x.ndim=}."
        num_rows = x.shape[0]
    elif isinstance(x, Sequence):
        num_rows = len(x)
    else:
        raise ValueError("Input must be at least 2D tensor or a sequence.")
    if with_replacement:
        perms_iter = itertools.product(range(num_rows), repeat=arity)
    else:
        perms_iter = itertools.permutations(range(num_rows), r=arity)
    batch = []
    for perm in perms_iter:
        permuted = sliceit(x, perm)
        batch.append(
            PermutedData(
                perm,
                transform(permuted),
            )
        )
        if batch_size is not None:
            if len(batch) == batch_size:
                yield yield_batch(batch)
                batch = []
    if batch:
        yield yield_batch(batch)


class SeparatingSequentialBatchSampler(torch.utils.data.BatchSampler):
    """
    A BatchSampler that separates batches for each dataset in a ConcatDataset.

    Useful for validation datasets where you want to keep the batches separated by dataset,
    but want to use a single DataLoader.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        batch_size: int,
        drop_last: bool = False,
    ):
        super().__init__(None, batch_size, drop_last)
        self.separation_indices = [0] + ConcatDataset.cumsum(datasets)[:-1]
        self.batch_samplers = [
            BatchSampler(
                SequentialSampler(dataset),
                batch_size=batch_size,
                drop_last=drop_last,
            )
            for dataset in datasets
        ]

    def __iter__(self):
        for start_index, batch_sampler in zip(
            self.separation_indices, self.batch_samplers, strict=True
        ):
            for batch in batch_sampler:
                yield [start_index + i for i in batch]

    def __len__(self):
        return sum(map(len, self.batch_samplers))


def expand_sequence(
    batch_size: Sequence[int],
    time_size: int,
    data: Sequence,
) -> list:
    """
    Expands a flat data sequence into a nested list of shape specified by `batch_size` such that
    the last dimension selects from `data` and each selected element is repeated `time_size` times.

    If batch_size is (B1, B2, ..., Bn), then the output is a nested list where
    ``out[b1][b2]...[bn] == [data[bn]] * time_size`` for each valid index.  The earlier dimensions
    only control the nesting depth and duplication of the deeper structure.
    """
    return _expand_sequence(batch_size, time_size, data, dim=0)


def _expand_sequence(batch_size, time_size, data, dim):
    size = batch_size[dim]
    if dim != len(batch_size) - 1:
        # If we are not at the last dimension, we need to recurse
        return [
            _expand_sequence(batch_size, time_size, data, dim=dim + 1)
            for _ in range(size)
        ]
    else:
        # If we are at the last dimension, we can directly create the relabelling instances
        return [[data[b]] * time_size for b in range(size)]
