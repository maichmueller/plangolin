import itertools
from dataclasses import dataclass
from typing import Callable, Generator, Generic, Sequence, TypeVar

import torch

try:
    from itertools import batched
except ImportError:
    from collections.abc import Iterable, Iterator
    from itertools import islice
    from typing import List, TypeVar

    U = TypeVar("U")

    def batched(iterable: Iterable[U], n: int) -> Iterator[List[U]]:
        """Batch data into lists of length n. The last batch may be shorter.

        Sufficiently equivalent to itertools.batched, available in Python 3.12+.
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


@dataclass(slots=True, frozen=True)
class PermutedDataBatch(Generic[T]):
    permutations: list[tuple[int, ...]]
    data: list[T | torch.Tensor]
    arity: int


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
        perms, data = zip(*map(lambda elem: (elem.permutation, elem.data), batch_))
        return PermutedDataBatch(perms, data, arity=arity)

    def sliceit(data: torch.Tensor | T, indices: Sequence[int]) -> T | torch.Tensor:
        if isinstance(x, torch.Tensor):
            return data[list(indices)]
        else:
            return [data[i] for i in indices]

    match x:
        case torch.Tensor():
            assert x.ndim >= 2, f"Input must be at least 2D tensor. Given {x.ndim=}."
        case Sequence():
            ...
        case _:
            raise ValueError("Input must be at least 2D tensor or a sequence.")
    num_rows = x.shape[0]
    if arity == 1:
        # no need to permute
        perms_iter = batched(range(x.shape[0]), batch_size or num_rows)
    else:
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
