from test.fixtures import *

import torch

from rgnet.models.attention_aggr import AttentionAggregation


def test_attention_aggr():
    # Create a random tensor to simulate node features
    x = torch.randn(10, 6)  # 10 nodes, 6 features each
    index = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3, 4, 4])  # two nodes are reduced to one
    assert len(x) == len(index), "Index length must match feature length"

    # Initialize the AttentionAggregation layer
    attention_aggr = AttentionAggregation(
        feature_size=6, num_heads=2, split_features=True
    )

    # Forward pass through the attention aggregation layer
    output = attention_aggr(x, index=index)

    # Check the output shape
    assert output.shape == (5, 6), "Output shape mismatch"

    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor"
