from test.fixtures import *  # noqa: F401, F403

import mockito
import pytest
import torch

from plangolin.rl.embedding import EmbeddingTransform, NonTensorTransformedEnv
from plangolin.rl.envs import ExpandedStateSpaceEnv
from plangolin.utils.object_embeddings import ObjectEmbedding

from .envs.test_state_space_env import get_expected_next_keys, get_expected_root_keys


def match_non_tensor_stack(expected_stack):
    """Because comparing NonTensorStacks might throw an exception about comparing
    tensors with more than one dimension."""

    def match(arg):
        return arg.tolist() == expected_stack.tolist()

    return mockito.arg_that(match)


@pytest.mark.parametrize("embedding_size", [3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_forward(small_blocks, embedding_mock, batch_size, embedding_size):
    """Verify that embeddings are produced in an environment step and reset operation."""
    space, _, _ = small_blocks
    env = ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)

    emb_transform = EmbeddingTransform(
        "current_embedding", env=env, embedding_module=embedding_mock
    )

    transformed = NonTensorTransformedEnv(env, emb_transform, cache_specs=True)
    td = transformed.reset()

    env_keys = env.keys
    expected_keys = get_expected_root_keys(env)
    expected_keys.append("current_embedding")
    expected_keys = sorted(expected_keys)
    assert td.sorted_keys == expected_keys
    mockito.verify(embedding_mock, times=1).forward(
        match_non_tensor_stack(td.get(env_keys.state))
    )
    current_embedding: ObjectEmbedding = ObjectEmbedding.from_tensordict(
        td["current_embedding"]
    )
    assert current_embedding.dense_embedding.shape == (
        batch_size,
        embedding_mock.test_num_objects,
        embedding_mock.embedding_size,
    )
    assert current_embedding.dense_embedding.requires_grad

    transformed.rand_step(td)
    expected_next_keys = sorted(
        [
            "current_embedding",
        ]
        + get_expected_next_keys(env)
    )
    assert td["next"].sorted_keys == expected_next_keys
    mockito.verify(embedding_mock, times=1).forward(
        match_non_tensor_stack(td.get(("next", env_keys.state)))
    )
    assert ObjectEmbedding.from_tensordict(
        td[("next", "current_embedding")]
    ).dense_embedding.requires_grad

    out = transformed._step_mdp(td)
    expected_t_1_keys = sorted(expected_keys + ["action"])

    assert out.sorted_keys == expected_t_1_keys
    mockito.verify(embedding_mock, times=2).forward(...)  # no new invocations
    assert ObjectEmbedding.from_tensordict(
        out["current_embedding"]
    ).dense_embedding.shape == torch.Size(
        [batch_size, embedding_mock.test_num_objects, embedding_mock.embedding_size]
    )
    assert ObjectEmbedding.from_tensordict(out["current_embedding"]).allclose(
        td[("next", "current_embedding")]
    )

    # assert that the spec complies with the tensordict using _StepMDP
    assert transformed._step_mdp.validate(td)


@pytest.mark.parametrize("embedding_size", [3])
@pytest.mark.parametrize("batch_size", [2])
def test_partial_reset(small_blocks, embedding_mock, batch_size, embedding_size):
    """
    Ensure that we do not produce embeddings for batch-entries that are not done,
    during a partial reset. Specifically when the _reset of the base_env is called in
    the process of a partial reset we do not want to compute unnecessary embeddings for
    the batch-entries that are not done.
    """
    space, _, _ = small_blocks
    env = ExpandedStateSpaceEnv(space, batch_size=torch.Size([batch_size]), seed=42)

    emb_transform = EmbeddingTransform(
        "current_embedding", env=env, embedding_module=embedding_mock
    )

    transformed = NonTensorTransformedEnv(env, emb_transform, cache_specs=True)
    td = transformed.reset()

    transformed.rand_step(td)
    mockito.verify(embedding_mock, times=2).forward(...)
    mockito.forget_invocations(embedding_mock)

    td[("next", "done")] = torch.tensor([True, False])
    tensordict_ = transformed._step_mdp(td)
    tensordict_ = transformed.maybe_reset(tensordict_)

    mockito.verify(embedding_mock, times=1).forward(...)

    # Assert that the not-done entry is still the same
    assert ObjectEmbedding.from_tensordict(
        tensordict_["current_embedding"][1]
    ).allclose(td[("next", "current_embedding")][1])

    assert not ObjectEmbedding.from_tensordict(
        tensordict_["current_embedding"][0]
    ).allclose(td[("next", "current_embedding")][0])
    assert tensordict_[env.keys.state][0] != td[("next", env.keys.state)][0]
    assert tensordict_[env.keys.state][1] == td[("next", env.keys.state)][1]
