from test.fixtures import (  # noqa: F401
    expanded_state_space_env,
    medium_blocks,
    request_accelerator_for_test,
)
from typing import List

import lightning
import mockito
import pytest
import torch
from tensordict import LazyStackedTensorDict, TensorDict
from torchrl.modules import ValueOperator

from rgnet.rl.non_tensor_data_utils import as_non_tensor_stack
from rgnet.rl.thundeRL.validation import CriticValidation, ProbsCollector


@pytest.mark.parametrize(
    "expanded_state_space_env", [["medium_blocks", 25]], indirect=True
)
def test_critic_validation(expanded_state_space_env):
    """
    Tests the critic validation callback.
    Go over whole state space in 5 batches, each 25 elements.
    The value operator is mocked and produces estimates one higher than optimal.
    The optimal values are just 0,...,124, but the indices of states are reversed.
    Therefore, the callback has to sort the predictions based on the idx_in_space.
    The expected loss is 1.0 = torch.nn.functional.mse([0,...,124],[1,...,125])
    """
    env = expanded_state_space_env
    batch_size = 25
    idx_in_space = torch.arange(start=124, end=-1, step=-1, dtype=torch.long)
    overestimated_values = torch.arange(start=125, end=0, step=-1, dtype=torch.float)

    class OverestimatingValueOperator(torch.nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.batch_index = 0

        def forward(self, states):
            values = overestimated_values[
                batch_size * self.batch_index : batch_size * (self.batch_index + 1)
            ]
            self.batch_index += 1
            return values.unsqueeze(dim=-1)

    optimal_values_dict = {0: torch.arange(0, 125, dtype=torch.float)}

    value_operator = ValueOperator(
        in_keys=["state"], module=OverestimatingValueOperator()
    )
    critic_validation = CriticValidation(
        discounted_optimal_values=optimal_values_dict, value_operator=value_operator
    )
    for batch_index in range(5):
        rollout = env.rollout(max_steps=1)
        rollout["idx_in_space"] = idx_in_space[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ].unsqueeze(dim=-1)
        critic_validation(rollout)

    d = dict()

    def log_mock(name, key, on_epoch: bool = True):
        d[name] = key

    pl_mock = mockito.mock({"log": log_mock}, spec=lightning.LightningModule)
    critic_validation.on_validation_epoch_end(
        mockito.mock(spec=lightning.Trainer), pl_mock
    )
    values = list(d.values())
    assert len(values) == 1
    loss = values[0]
    assert loss == pytest.approx(1.0)


def tensor_list_eq(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    t1: torch.Tensor
    return all((t1 == t2).all() for (t1, t2) in zip(l1, l2))


class TestProbsCollector:
    device = request_accelerator_for_test("TestProbsCollector")

    @pytest.fixture
    def collector(self) -> ProbsCollector:
        # noinspection PyTypeChecker
        return ProbsCollector(probs_key="probs")

    @pytest.fixture
    def sample_data(self) -> List[TensorDict]:
        raw_data = [
            (
                [
                    torch.tensor([0.3, 0.7], device=self.device),
                    torch.tensor([1.0], device=self.device),
                ],
                torch.tensor([0, 1], device=self.device),
            ),
            (
                [
                    torch.tensor([0.4, 0.6], device=self.device),
                    torch.tensor([0.25, 0.25, 0.5], device=self.device),
                ],
                torch.tensor([3, 2], device=self.device),
            ),
        ]
        tds = [
            TensorDict(
                {"probs": as_non_tensor_stack(probs), "idx_in_space": idx_in_space},
                device=self.device,
                batch_size=(2,),
            )
            for (probs, idx_in_space) in raw_data
        ]
        return [
            LazyStackedTensorDict.maybe_dense_stack(
                [td], len(td.batch_size)
            ).refine_names(..., "time")
            for td in tds
        ]

    @pytest.fixture
    def expected_sorted_probs(self) -> List[torch.Tensor]:
        return [
            torch.tensor(ls, device=self.device)
            for ls in ([0.3, 0.7], [1.0], [0.25, 0.25, 0.5], [0.4, 0.6])
        ]

    # Test case 1: Basic functionality
    def test_basic_functionality(self, collector, sample_data, expected_sorted_probs):

        # Test forward pass
        collector.forward(sample_data[0], batch_idx=0)

        # Verify storage
        epoch_probs: List[torch.Tensor] = collector.sort_probs_on_epoch_end()[0]
        assert isinstance(epoch_probs, List)
        assert all(isinstance(tensor, torch.Tensor) for tensor in epoch_probs)
        assert len(epoch_probs) == 2
        assert epoch_probs[0].ndim == 1
        assert all(t.device == self.device for t in epoch_probs)
        assert tensor_list_eq(epoch_probs, expected_sorted_probs[:2])
        assert len(collector.sort_probs_on_epoch_end().keys()) == 1

    # Test case 2: Multiple batches
    def test_multiple_batches(self, collector, sample_data, expected_sorted_probs):

        # Create multiple batches
        for batch_idx, td in enumerate(sample_data):
            collector.forward(td, batch_idx=batch_idx)

        # Test sorting
        epoch_probs: List[torch.Tensor] = collector.sort_probs_on_epoch_end()[0]
        assert tensor_list_eq(epoch_probs, expected_sorted_probs)
        assert len(collector.sort_probs_on_epoch_end().keys()) == 1

    # Test case 3: Duplicate batch detection
    def test_duplicate_batch(self, collector, sample_data, expected_sorted_probs):

        td = sample_data[0]

        # Submit the same batch twice
        collector.forward(td, batch_idx=0)
        collector.forward(td, batch_idx=0)

        # Verify no duplication
        epoch_probs: List[torch.Tensor] = collector.sort_probs_on_epoch_end()[0]
        assert len(epoch_probs) == 2  # first batch contains two elements
        assert tensor_list_eq(epoch_probs, expected_sorted_probs[:2])

    # Test case 4: Multiple dataloaders
    def test_multiple_dataloaders(self, collector, sample_data, expected_sorted_probs):

        # Submit to different dataloader indices
        for dataloader_idx in range(2):
            for idx, td in enumerate(sample_data):
                collector.forward(td, batch_idx=idx, dataloader_idx=dataloader_idx)

        # Verify data in each dataloader
        epoch_probs_dict = collector.sort_probs_on_epoch_end()
        assert set(epoch_probs_dict.keys()) == {0, 1}
        for epoch_probs in epoch_probs_dict.values():
            assert tensor_list_eq(epoch_probs, expected_sorted_probs)

    def test_empty_epoch(self, collector):
        # Test getting sorted probs without any data
        sorted_probs = collector.sort_probs_on_epoch_end()
        assert isinstance(sorted_probs, dict)
        assert len(sorted_probs) == 0

    def test_duplicate_batch_different_dataloader(
        self, collector, sample_data, expected_sorted_probs
    ):

        td0, td1 = sample_data

        # Submit the same batch twice but under different dataloader
        collector.forward(td0, batch_idx=0)
        collector.forward(td1, batch_idx=0, dataloader_idx=1)

        epoch_probs_dict = collector.sort_probs_on_epoch_end()
        epoch_probs0, epoch_probs1 = epoch_probs_dict[0], epoch_probs_dict[1]
        assert len(epoch_probs0) == 2
        assert len(epoch_probs1) == 2
        assert tensor_list_eq(epoch_probs0, expected_sorted_probs[:2])
        assert tensor_list_eq(epoch_probs1, expected_sorted_probs[2:])

    def test_empty_after_reset(self, collector, sample_data):
        # First add some data
        collector.forward(sample_data[0], batch_idx=0)

        # Verify data is present
        assert len(collector.sort_probs_on_epoch_end().items()) > 0

        # Reset collector
        collector.reset()

        # Verify all data is cleared
        assert len(collector.sort_probs_on_epoch_end().items()) == 0

    # Test that seen batch indices are reset correctly
    def test_reset(self, collector, sample_data, expected_sorted_probs):

        collector.forward(sample_data[0], batch_idx=0)

        collector.reset()

        collector.forward(sample_data[1], batch_idx=0)

        epoch_probs: List[torch.Tensor] = collector.sort_probs_on_epoch_end()[0]

        assert tensor_list_eq(epoch_probs, expected_sorted_probs[2:])

    def test_invalid_input(self, collector):
        # Test with mismatched sizes
        invalid_probs = as_non_tensor_stack(
            [[torch.tensor([0.3, 0.7], device=self.device)]]
        )
        invalid_state_idx = torch.tensor(
            [0, 1], device=self.device
        )  # Two indices but only one prob
        td = TensorDict({"probs": invalid_probs, "idx_in_space": invalid_state_idx})

        with pytest.raises(
            ValueError,
            match="Number of probability tensors must match number of states",
        ):
            collector.forward(td, batch_idx=0)
