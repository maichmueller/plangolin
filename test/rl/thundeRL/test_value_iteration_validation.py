import mockito
import pymimir as mi
import pytest
import torch
from fixtures import medium_blocks
from mockito import arg_that, mock, spy2, verify, when
from tensordict import TensorDict
from torch import Tensor

from rgnet.rl.thundeRL.validation import ValueIterationValidation


class TestValueIterationValidation:

    @pytest.fixture
    def mock_probs_collector(self):
        collector = mock({"__call__": lambda *args, **kwargs: None}, strict=True)
        when(collector).reset()
        return collector

    @pytest.fixture
    def optimal_values(self):
        # Create mock optimal values for one validation space
        return {0: torch.tensor([-1.0, -0.8, -0.6])}

    def test_initialization(self, medium_blocks, mock_probs_collector, optimal_values):
        space_that_not_be_used = mock(spec=mi.StateSpace, strict=True)
        medium_space = medium_blocks[0]
        spaces = [medium_space, space_that_not_be_used]
        validator = ValueIterationValidation(
            spaces=spaces,
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            gamma=0.99,
            num_iterations=10,
            only_run_for_dataloader={0},
        )

        assert len(validator._graphs) == 1
        assert validator.message_passing.gamma == 0.99
        assert validator.message_passing.num_iterations == 10

    def test_initialization_validation_error(
        self, medium_blocks, mock_probs_collector, optimal_values
    ):
        with pytest.raises(ValueError) as exc_info:
            ValueIterationValidation(
                spaces=medium_blocks[0],
                discounted_optimal_values=optimal_values,
                probs_collector=mock_probs_collector,
                gamma=0.99,
                num_iterations=None,
                difference_threshold=None,
            )
        assert "Neither num_iterations nor difference_threshold was given" in str(
            exc_info.value
        )

    def test_forward(self, medium_blocks, mock_probs_collector, optimal_values):
        validator = ValueIterationValidation(
            spaces=[medium_blocks[0]],
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            gamma=0.99,
            num_iterations=10,
            only_run_for_dataloader={0},
        )
        td = mock(spec=TensorDict)
        validator(td, 0, 0)
        verify(mock_probs_collector, times=1).__call__(
            td, batch_idx=0, dataloader_idx=0
        )
        mockito.forget_invocations(mock_probs_collector)
        validator(td, 0, 1)
        mockito.verifyZeroInteractions(mock_probs_collector)

    def test_validation_epoch_start(
        self, medium_blocks, mock_probs_collector, optimal_values
    ):
        spaces = [medium_blocks[0]]
        validator = ValueIterationValidation(
            spaces=spaces,
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            gamma=0.99,
            num_iterations=10,
        )

        mock_trainer = mock(strict=True)
        mock_pl_module = mock(strict=True)

        validator.on_validation_epoch_start(mock_trainer, mock_pl_module)
        verify(mock_probs_collector, times=1).reset()

    def test_compute_values_error(
        self, medium_blocks, mock_probs_collector, optimal_values
    ):
        space = medium_blocks[0]
        validator = ValueIterationValidation(
            spaces=[space],
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            gamma=0.99,
            num_iterations=10,
            only_run_for_dataloader={0},
        )
        num_transitions = [
            len(space.get_forward_transitions(s)) for s in space.get_states()
        ]
        # Test bad probs that don't match state space
        with pytest.raises(AssertionError):
            bad_probs = [torch.zeros((2,))] * space.num_states()
            validator.compute_values(bad_probs, 0)
        # Test bad probs of correct form but with extra dimension
        with pytest.raises(AssertionError):
            bad_probs = [
                torch.zeros((num_t,)).unsqueeze(dim=-1) for num_t in num_transitions
            ]
            validator.compute_values(bad_probs, 0)

        # Test bad probs on the wrong device
        with pytest.raises(AssertionError):
            bad_probs = [
                torch.zeros((num_t,), device=torch.device("meta"))
                for num_t in num_transitions
            ]
            validator.compute_values(bad_probs, 0)

    def test_compute_values(self, medium_blocks, mock_probs_collector, optimal_values):
        space = medium_blocks[0]
        spaces = [space]
        validator = ValueIterationValidation(
            spaces=spaces,
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            gamma=0.99,
            num_iterations=10,
            only_run_for_dataloader={0},
        )
        num_transitions = [
            len(space.get_forward_transitions(s)) for s in space.get_states()
        ]
        probs = [torch.rand((num_t,)).softmax(dim=-1) for num_t in num_transitions]
        spy2(validator.message_passing.forward)
        validator.compute_values(probs, 0)
        verify(validator.message_passing).forward(
            arg_that(lambda data: (data.edge_attr[:, 0] == torch.cat(probs)).all())
        )

    def test_validation_epoch_end(
        self, medium_blocks, mock_probs_collector, optimal_values
    ):
        spaces = [medium_blocks[0]]
        validator = ValueIterationValidation(
            spaces=spaces,
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            gamma=0.99,
            num_iterations=10,
            only_run_for_dataloader={0},
        )

        mock_trainer = mock(strict=True)
        mock_pl_module = mock(strict=True)

        # Mock the collected probabilities for the epoch
        num_transitions = [
            len(spaces[0].get_forward_transitions(state))
            for state in spaces[0].get_states()
        ]
        collected_probs = [
            torch.rand((num_trans,)).softmax(dim=-1) for num_trans in num_transitions
        ]
        sorted_epoch_probs = {
            0: collected_probs,
            1: None,
            # ProbsCollector could be shared. ValueIterationValidation has to filter!
        }
        mock_probs_collector.sorted_epoch_probs = sorted_epoch_probs

        # Mock the compute_values method to return a tensor of the same shape as optimal_values
        mp_return: Tensor = optimal_values[0] + 1.0
        when(validator).compute_values(collected_probs, 0).thenReturn(mp_return)

        # The Expected mse_loss is 1.0, every value is one greater than optimal
        when(mock_pl_module).log(validator.log_key(0), 1.0, on_epoch=True)

        validator.on_validation_epoch_end(mock_trainer, mock_pl_module)

        # ProbsCollector stores probs on the wrong device should trigger assertion
        with pytest.raises(AssertionError):
            meta_device = torch.device("meta")
            collected_probs = [t.to(meta_device) for t in collected_probs]
            sorted_epoch_probs[0] = collected_probs
            when(validator).compute_values(...).thenReturn(
                optimal_values[0].to(meta_device)
            )
            validator.on_validation_epoch_end(mock_trainer, mock_pl_module)
