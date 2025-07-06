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
    Yields batches of all permutations of rows (dim=0) of a 2D tensor or 1D sequences.

    Args:
        x: torch.Tensor,
            A 2D tensor (rows will be permuted) or 1D tensor sequence of equal length.
        arity: int,
            Number of objects per permutation.
        batch_size: int,
            Number of permutations per batch.
        transform: Callable,
            A function to transform the data before yielding it.
        with_replacement: bool,
            If True, allows for repeated elements in the permutations.

    Yields:
        torch.Tensor: A tensor of shape (≤batch_size, rows, cols).
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
