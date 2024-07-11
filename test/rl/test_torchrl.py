from typing import Optional

import pytest
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import BinaryDiscreteTensorSpec, CompositeSpec, NonTensorSpec
from torchrl.envs import EnvBase, step_mdp
from torchrl.envs.utils import _update_during_reset
from torchrl.objectives.value import TD0Estimator

from rgnet.rl import torchrl_patches

print(torchrl_patches)


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


def test_value_estimators_td0():

    prediction = torch.randn(size=(2, 5, 1))
    next_prediction = prediction[:, :5, :]

    gamma = 0.9

    value_network = TensorDictModule(
        lambda x: 1 / 0,  # raise exception
        in_keys=["obs"],
        out_keys=["state_value"],
    )
    estimator = TD0Estimator(
        gamma=gamma,
        value_network=value_network,
        average_rewards=False,
        skip_existing=True,
    )

    td = TensorDict(
        {
            # "obs": torch.empty(size=(2, 5, 1)),
            "state_value": prediction,
            ("next", "state_value"): next_prediction,
            # ("next", "obs"): torch.empty(size=(2, 5, 1)),
            ("next", "reward"): torch.ones(size=(2, 5, 1)),
            ("next", "done"): torch.zeros((2, 5, 1), dtype=torch.bool),
        },
        batch_size=(2, 5),
    )
    # will calculate TD0 for each time step independently
    estimator(td)
    # td advantage is R_{t+1} + gamma * V(S_{t+1}) - V(S_t)
    # td value_target is V(S_t) + advantage
    expected_advantage = td[("next", "reward")] + gamma * next_prediction - prediction
    expected_value_target = prediction + expected_advantage
    assert "value_target" in td
    assert "advantage" in td
    assert torch.allclose(expected_advantage, td["advantage"])
    assert torch.allclose(expected_value_target, td["value_target"])


def test_non_tensor_data():
    td = TensorDict({"x": torch.tensor([1.0, 2.0, 3.0])}, batch_size=(3,))

    TensorDictModule(
        lambda: NonTensorData(["a", "b", "c"]), in_keys=[], out_keys=["y"]
    )(td)

    assert td["y"] == ["a", "b", "c"]


def test_non_tensor_stack_ragged_tensors():
    t1 = torch.tensor([1, 2, 3], dtype=torch.float)
    t2 = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    stack = NonTensorStack(NonTensorData(t1), NonTensorData(t2))
    assert all(isinstance(t, torch.Tensor) for t in stack.tolist())
    stack = torch.stack([NonTensorData(t1), NonTensorData(t2)])
    assert all(isinstance(t, torch.Tensor) for t in stack.tolist())


def test_environment_partial_reset():
    """Test partial reset bahvior with non tensor data.
    See isesue https://github.com/pytorch/rl/issues/2257 for more details.
    """
    test_batch_size = 2

    class CustomEnv(EnvBase):
        # Custom environment

        def __init__(
            self,
            *,
            device=None,
            batch_size: Optional[torch.Size] = torch.Size([test_batch_size]),
            run_type_checks: bool = False,
            allow_done_after_reset: bool = False,
        ):
            assert batch_size == (test_batch_size,)  # hardcoded for minimal example
            super().__init__(
                device=device,
                batch_size=batch_size,
                run_type_checks=run_type_checks,
                allow_done_after_reset=allow_done_after_reset,
            )

            self.observation_spec = CompositeSpec(
                observation=NonTensorSpec(shape=batch_size), shape=batch_size
            )
            self.action_spec = NonTensorSpec(shape=batch_size)
            self.reward_spec: BinaryDiscreteTensorSpec = BinaryDiscreteTensorSpec(
                n=1, dtype=torch.int8, shape=torch.Size([test_batch_size, 1])
            )

        def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
            done = torch.tensor([True, False], dtype=torch.bool)
            next_observation = NonTensorStack(
                NonTensorData("B"), NonTensorData("Z"), batch_size=(test_batch_size,)
            )
            return TensorDict(
                {
                    "observation": next_observation,
                    "done": done,
                    "reward": torch.ones((test_batch_size,)),
                },
                batch_size=(test_batch_size,),
            )

        def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
            return TensorDict(
                {
                    "observation": NonTensorStack(
                        NonTensorData("A"),
                        NonTensorData("C"),
                        batch_size=(test_batch_size,),
                    )
                },
                batch_size=(test_batch_size,),
            )

        def rand_action(self, tensordict: Optional[TensorDictBase] = None):
            action = NonTensorStack(
                *([NonTensorData("+")] * test_batch_size), batch_size=(test_batch_size,)
            )
            if tensordict is None:
                tensordict = TensorDict({}, batch_size=self.batch_size)
            tensordict["action"] = action
            return tensordict

        def _set_seed(self, seed: Optional[int]):
            pass

    env = CustomEnv()
    td = env.reset()
    env.rand_action(td)
    out_td, reset_td = env.step_and_maybe_reset(td)
    assert out_td is td
    assert torch.equal(td["next", "done"], torch.tensor([[True], [False]]))
    observation = "observation"
    next_observation = ("next", observation)

    assert td[next_observation] == ["B", "Z"]  # the result of _step
    assert reset_td[observation] == ["A", "Z"]  # first entry is reset
