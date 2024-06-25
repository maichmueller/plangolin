import pytest
import torch
from tensordict import LazyStackedTensorDict, NonTensorData, NonTensorStack, TensorDict
from torchrl.envs import step_mdp
from torchrl.envs.utils import _update_during_reset

from rgnet.rl import torchrl_patches

print(torchrl_patches)  # leaving this in so the import is not optimized


def test_step_function():
    """Test whether NonTensorStack entries get correctly moved in step_mdp"""
    now = NonTensorStack(*[NonTensorData("a"), NonTensorData("b")], batch_size=(2,))
    next_ = NonTensorStack(*[NonTensorData("c"), NonTensorData("d")], batch_size=(2,))
    action = NonTensorStack(*[NonTensorData("1"), NonTensorData("2")])
    reward = torch.ones(size=(2,))
    done = torch.ones(size=(2,), dtype=torch.bool)
    terminated = done.clone()
    td = TensorDict(
        {
            "x": now,
            "done": torch.zeros(size=(2,), dtype=torch.bool),
            "terminated": terminated,
            ("next", "x"): next_,
            ("next", "done"): done,
            ("next", "reward"): reward,
            ("next", "terminated"): terminated,
            ("next", "action"): action,
        },
        batch_size=(2,),
    )
    next_td = step_mdp(
        td,
        keep_other=True,
        exclude_reward=False,
        exclude_action=False,
        exclude_done=False,
    )

    assert next_td.batch_size == (2,)
    expected_keys = ["action", "done", "reward", "terminated", "x"]
    assert expected_keys == next_td.sorted_keys
    assert "next" not in next_td.keys(include_nested=True)
    assert next_td.get("x") is next_
    assert next_td.get("done") is done
    assert next_td.get("reward") is reward
    assert next_td.get("terminated") is terminated
    assert next_td.get("action") is action


def test_where_with_non_tensor_stack():
    condition = torch.tensor([True, False])
    tensor = NonTensorStack(
        *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
    )
    other = NonTensorStack(
        *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
    )
    out = NonTensorStack(*[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,))
    result = tensor.where(condition=condition, other=other, out=out, pad=0)
    assert isinstance(result, NonTensorStack)

    # Underlying problem
    # LazyStackedTensorDict.maybe_dense_stack(
    #    [
    #        td.where(cond, _other, pad=0)
    #        for td, cond, _other in zip(tensor.tensordicts, condition, other)
    #    ],
    #    tensor.stack_dim,  # == 0
    # )


def test_stack_non_tensor_data():
    data = torch.stack(
        [
            NonTensorData("a"),
            NonTensorData("a"),
        ]
    )
    stack = NonTensorStack(
        *(NonTensorData("b"), NonTensorData("b")),
    )
    assert torch.stack([data, stack], dim=1).batch_size == (2, 2)


# Problem is not yet fixed torch.cat not implemented for NonTensorData and NonTensorStack
@pytest.mark.skip
def test_cat_non_tensor_data():
    data = torch.cat(
        [
            NonTensorData("a", batch_size=(1,)),
            NonTensorData("a", batch_size=(1,)),
        ]
    )
    stack = NonTensorStack(
        *(NonTensorData("b"), NonTensorData("b")),
        batch_size=(2,),
    )
    assert torch.cat([data, stack], dim=0).batch_size == (4,)


def test_update_during_reset():
    tensordict_reset = TensorDict(
        {"observation": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        batch_size=(2,),
    )
    tensordict = TensorDict(
        {
            "_reset": torch.tensor([True, False], dtype=torch.bool),
            "observation": torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]),
        },
        batch_size=(2,),
    )
    _update_during_reset(tensordict_reset, tensordict, ["_reset"])
    # We expect that the first entry is replaced by teh reset tensordict
    assert torch.allclose(
        tensordict["observation"], torch.tensor([[1.0, 2.0], [-3.0, -4.0]])
    )
