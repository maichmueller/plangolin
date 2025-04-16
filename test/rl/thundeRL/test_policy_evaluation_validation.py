from test.fixtures import (  # noqa: F401, F403
    medium_blocks,
    request_accelerator_for_test,
)

import mockito
import pytest
import torch
from mockito import arg_that, mock, spy2, verify, when
from tensordict import TensorDict
from torch import Tensor

from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.reward import UnitReward
from rgnet.rl.thundeRL.validation import PolicyEvaluationValidation
from xmimir.iw import IWSearch, IWStateSpace


class TestPolicyEvaluationValidation:
    device = request_accelerator_for_test("TestPolicyEvaluationValidation")

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
        env_that_not_be_used = mock(spec=ExpandedStateSpaceEnv, strict=True)
        medium_space = medium_blocks[0]
        envs = [
            ExpandedStateSpaceEnv(
                medium_space, 1, reward_function=UnitReward(gamma=0.99), reset=True
            ),
            env_that_not_be_used,
        ]
        envs[0].reset()  # we need only the first space
        validator = PolicyEvaluationValidation(
            envs=envs,
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            num_iterations=10,
            log_aggregated_metric=False,
            only_run_for_dataloader={0},
        )

        assert len(validator._graphs) == 1
        assert all(
            message_passer.gamma == 0.99
            for message_passer in validator.message_passing.values()
        )
        assert all(
            message_passer.num_iterations == 10
            for message_passer in validator.message_passing.values()
        )

    def test_initialization_validation_error(
        self, medium_blocks, mock_probs_collector, optimal_values
    ):
        with pytest.raises(ValueError) as exc_info:
            PolicyEvaluationValidation(
                envs=[
                    ExpandedStateSpaceEnv(
                        medium_blocks[0],
                        reward_function=UnitReward(gamma=0.99),
                        reset=True,
                    )
                ],
                discounted_optimal_values=optimal_values,
                probs_collector=mock_probs_collector,
                num_iterations=None,
                log_aggregated_metric=False,
                difference_threshold=None,
            )
        assert "Neither num_iterations nor difference_threshold was given" in str(
            exc_info.value
        )

    def test_forward(self, medium_blocks, mock_probs_collector, optimal_values):
        validator = PolicyEvaluationValidation(
            envs=[
                ExpandedStateSpaceEnv(
                    medium_blocks[0], reward_function=UnitReward(gamma=0.99), reset=True
                )
            ],
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            num_iterations=10,
            log_aggregated_metric=False,
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
        validator = PolicyEvaluationValidation(
            envs=[
                ExpandedStateSpaceEnv(
                    space, reward_function=UnitReward(gamma=0.99), reset=True
                )
                for space in spaces
            ],
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            num_iterations=10,
            log_aggregated_metric=False,
        )

        mock_trainer = mock(strict=True)
        mock_pl_module = mock(strict=True)

        validator.on_validation_epoch_start(mock_trainer, mock_pl_module)
        verify(mock_probs_collector, times=1).reset()

    def test_compute_values_error(
        self, medium_blocks, mock_probs_collector, optimal_values
    ):
        space = medium_blocks[0]
        validator = PolicyEvaluationValidation(
            envs=[
                ExpandedStateSpaceEnv(
                    space, reward_function=UnitReward(gamma=0.99), reset=True
                )
            ],
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            num_iterations=10,
            log_aggregated_metric=False,
            only_run_for_dataloader={0},
        )
        num_transitions = [space.forward_transition_count(s) for s in space]
        # Test bad probs that don't match state space
        with pytest.raises(AssertionError):
            bad_probs = [torch.zeros((2,))] * len(space)
            validator.compute_values(bad_probs, 0)
        # Test bad probs of correct form but with extra dimension
        with pytest.raises(AssertionError):
            bad_probs = [
                torch.zeros((num_t,)).unsqueeze(dim=-1) for num_t in num_transitions
            ]
            validator.compute_values(bad_probs, 0)

        if self.device == torch.device("cpu"):
            pytest.skip("Skipping device sensible test")
        # Test bad probs on the wrong device
        with pytest.raises(AssertionError):
            bad_probs = [
                torch.zeros((num_t,), device=self.device) for num_t in num_transitions
            ]
            validator.compute_values(bad_probs, 0)

    @pytest.mark.parametrize("width", [0, 1])
    def test_compute_values(
        self, medium_blocks, width, mock_probs_collector, optimal_values
    ):
        """
        Test compute_values e.g., the internal pass to the message passing module.
        The function should return the computed values und the provided policy.
        We assert that the message passing module is called with the correct parameters.
        """
        space = medium_blocks[0]
        if width > 0:
            space = IWStateSpace(IWSearch(width), space)
        env = ExpandedStateSpaceEnv(
            space, reward_function=UnitReward(gamma=0.99), reset=True
        )
        validator = PolicyEvaluationValidation(
            envs=[env],
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            num_iterations=10,
            log_aggregated_metric=False,
            only_run_for_dataloader={0},
        )
        num_transitions = [len(env.get_applicable_transitions([s])[0]) for s in space]
        probs = [torch.rand((num_t,)).softmax(dim=-1) for num_t in num_transitions]
        spy2(validator.message_passing[0].forward)
        validator.compute_values(probs, 0)
        verify(validator.message_passing[0]).forward(
            arg_that(lambda data: (data.edge_attr[:, 0] == torch.cat(probs)).all())
        )

    def test_validation_epoch_end(
        self, medium_blocks, mock_probs_collector, optimal_values
    ):
        """
        Test on_validation_epoch_end with valid and invalid data.
        All other side effects are mocked:
            compute_values, ProbsCollector, Trainer and PolicyGradientModule.
        """
        space = medium_blocks[0]
        validator = PolicyEvaluationValidation(
            envs=[
                ExpandedStateSpaceEnv(
                    space, reward_function=UnitReward(gamma=0.99), reset=True
                )
            ],
            discounted_optimal_values=optimal_values,
            probs_collector=mock_probs_collector,
            num_iterations=10,
            log_aggregated_metric=True,
            only_run_for_dataloader={0},
        )

        mock_trainer = mock(strict=True)
        mock_pl_module = mock(strict=True)

        # Mock the collected probabilities for the epoch
        num_transitions = [space.total_transition_count]
        collected_probs = [
            torch.rand((num_trans,)).softmax(dim=-1) for num_trans in num_transitions
        ]
        sorted_epoch_probs = {
            0: collected_probs,
            # ProbsCollector could be shared. PolicyEvaluationValidation has to filter!
            1: None,
        }
        when(mock_probs_collector).sort_probs(...).thenReturn(collected_probs)
        when(mock_probs_collector).sort_probs_on_epoch_end(...).thenReturn(
            sorted_epoch_probs
        )

        # Mock the compute_values method to return a tensor of the same shape as optimal_values
        mp_return: Tensor = optimal_values[0] + 1.0
        when(validator).compute_values(collected_probs, 0).thenReturn(mp_return)

        # The Expected mse_loss is 1.0, every value is one greater than optimal
        when(mock_pl_module).log(validator.log_key(0), 1.0, on_epoch=True)

        # We only have one evaluation problem so the aggregated value should be the same.
        when(mock_pl_module).log(f"val/{validator.log_name}", 1.0, on_epoch=True)

        validator.on_validation_epoch_end(mock_trainer, mock_pl_module)

        if self.device == torch.device("cpu"):
            pytest.skip("Skipping device sensible test")
        # ProbsCollector stores probs on the wrong device should trigger assertion
        with pytest.raises(AssertionError):
            meta_device = self.device
            collected_probs = [t.to(meta_device) for t in collected_probs]
            sorted_epoch_probs[0] = collected_probs
            validator._losses.clear()  # necessary or the cached loss will simply avoid our error setup
            when(validator).compute_values(...).thenReturn(
                optimal_values[0].to(meta_device)
            )
            validator.on_validation_epoch_end(mock_trainer, mock_pl_module)
