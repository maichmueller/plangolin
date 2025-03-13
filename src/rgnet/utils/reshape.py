from torch import Tensor, broadcast_tensors


def unsqueeze_right(x: Tensor, n: int) -> Tensor:
    """
    Insert `n` dimensions from the right.

    For example, for a tensor with shape (4,3,5) and n=2, this results in a tensor with shape (4,3,5,1,1)

    :param: x
        tensor to unsqueeze
    :param: n
        number of times to unsqueeze
    :return: unsqueezed tensor (view)
    """
    return x.reshape(*x.shape, *(n * (1,)))


def unsqueeze_right_like(x: Tensor, y: Tensor) -> Tensor:
    """
    Match the number of dimensions of `x` to `y` by unsqueezing on the right.

    For example, for a tensor `x` with shape (4,3,5) and a tensor `y` with shape (4,3,5,1,1),
    this results in a tensor with shape (4,3,5,1,1).

    :param: x
        tensor to unsqueeze
    :param: n
        number of times to unsqueeze
    :return: unsqueezed tensor (view)
    """
    return unsqueeze_right(x, y.ndim - x.ndim)


def unsqueeze_left(x: Tensor, n: int) -> Tensor:
    """
    Insert `n` dimensions from the left.

    For example, for a tensor with shape (4,3,5) and n=2, this results in a tensor with shape (1,1,4,3,5)

    :param: x
        tensor to unsqueeze
    :param: n
        number of times to unsqueeze
    :return: unsqueezed tensor (view)
    """
    return x.reshape(*(n * (1,)), *x.shape)


def unsqueeze_left_like(x: Tensor, y: Tensor) -> Tensor:
    """
    Match the number of dimensions of `x` to `y` by unsqueezing on the left.

    For example, for a tensor `x` with shape (4,3,5) and a tensor `y` with shape (1,1,4,3,5),
    this results in a tensor with shape (1,1,4,3,5).

    :param: x
        tensor to unsqueeze
    :param: n
        number of times to unsqueeze
    :return: unsqueezed tensor (view)
    """
    return unsqueeze_left(x, y.ndim - x.ndim)


def broadcast_right(*x: Tensor) -> tuple[Tensor, ...]:
    """
    Broadcasting on the right.

    Given multiple tensors, apply broadcasting with unsqueezed on the right.
    First, tensors are unsqueezed on the right to the same number of dimensions.
    Then, torch.broadcasting is used.

    Example:
        tensors with shapes (1,2,3), (1,2), (2)
        results in tensors with shape (2,2,3)
    Parameters
    ----------
    x
        tensors to broadcast
    Returns
    -------
        broadcasted tensors (views)
    """
    max_dim = max(el.ndim for el in x)
    unsqueezed = broadcast_tensors(
        *(unsqueeze_right(el, max_dim - el.ndim) for el in x)
    )
    return unsqueezed
