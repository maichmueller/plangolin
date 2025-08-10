import pytest
import torch

from plangolin.models.pooling import (
    GlobalAddPool,
    GlobalAttentionPool,
    GlobalMaxPool,
    GlobalMeanPool,
    GlobalSortPool,
)


@pytest.mark.parametrize(
    "pooling",
    [
        GlobalAddPool(),
        GlobalMeanPool(),
        GlobalMaxPool(),
        GlobalAttentionPool(feature_size=3, num_heads=1),
        GlobalSortPool(
            k=1
        ),  # only k=1 will pass this test, as k>1 concatenates the top k nodes of each graph as a single embedding
    ],
)
def test_pooling_shapes_and_api(pooling):
    # Create fake object embeddings: 5 objects, embedding size 3
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    # Batch vector: objects 0-2 -> graph 0, objects 3-4 -> graph 1
    batch = torch.tensor([0, 0, 0, 1, 1])
    batch_size = batch.max().item() + 1
    feature_size = x.size(1)

    for bs in [2, None]:
        result = pooling(x, batch, dim_size=bs)
        # Should return one embedding per unique graph in batch
        assert result.shape == (batch_size, feature_size)


@pytest.mark.parametrize(
    "pooling,expected",
    [
        (
            GlobalAddPool,
            torch.tensor([[12.0, 15.0, 18.0], [3.0, 3.0, 3.0]]),
        ),  # sum per graph
        (
            GlobalMeanPool,
            torch.tensor([[4.0, 5.0, 6.0], [1.5, 1.5, 1.5]]),
        ),  # average per graph
        (
            GlobalMaxPool,
            torch.tensor([[7.0, 8.0, 9.0], [2.0, 2.0, 2.0]]),
        ),  # max per graph
    ],
)
def test_pooling_correctness(pooling, expected):
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1])

    pooling_module = pooling()

    result = pooling_module(x, batch)
    assert torch.allclose(result, expected)


def test_pooling_empty_input():
    # Empty object embedding
    x = torch.empty((0, 3))
    batch = torch.empty((0,), dtype=torch.long)

    pooling_module = GlobalAddPool()

    result = pooling_module(x, batch)
    assert result.shape == (0, 3)
    assert result.numel() == 0


def test_pooling_single_element_batch():
    x = torch.tensor([[5.0, 5.0, 5.0]])
    batch = torch.tensor([0])

    pooling_module = GlobalMeanPool()

    result = pooling_module(x, batch)
    assert torch.equal(result, x)
