import itertools
import logging
import shutil
import sys
from pathlib import Path
from test.fixtures import medium_blocks, small_blocks  # noqa: F401, F403
from typing import Any, List, Tuple

import mockito
import pytest
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch_geometric.data import Batch, HeteroData

import rgnet
from rgnet.rl.data import FlashDrive
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.thundeRL import PolicyGradientCLI

from ..supervised.test_data import hetero_data_equal


@pytest.fixture(autouse=True, scope="class")
def setup_multiprocessing():
    torch.multiprocessing.set_start_method("fork", force=True)


def cli_main():
    logging.getLogger().setLevel(logging.INFO)
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")
    cli = PolicyGradientCLI()
    return cli


class PolicyGradientLitModuleMock:
    def __init__(self):
        super().__init__()
        self.batched_list: List[Batch] = []
        self.successor_batch_list: List[Batch] = []
        self.num_successor_list: List[torch.Tensor] = []

    def training_step_mock(
        self, batch_tuple: Tuple[Batch, Batch, torch.Tensor], **kwargs: Any
    ) -> STEP_OUTPUT:
        # We can no longer check for tuple exclusively because if a pytorch dataloader has pin_memory=True set,
        # their data will be converted to a list (https://github.com/pytorch/pytorch/issues/48419)
        assert isinstance(batch_tuple, (list, tuple))
        assert len(batch_tuple) == 4
        assert isinstance(batch_tuple[0], Batch)
        assert isinstance(batch_tuple[1], Batch)
        assert type(batch_tuple[2]) == torch.Tensor
        batched, successor_batch, num_successor, _ = batch_tuple
        assert batched.batch_size == num_successor.numel()
        assert batched.done.device == batched.reward.device == num_successor.device
        assert (
            num_successor.sum(dim=0)
            == successor_batch.batch_size
            == batched.done.numel()
            == batched.reward.numel()
        )
        self.batched_list.append(batched)
        self.successor_batch_list.append(successor_batch)
        self.num_successor_list.append(num_successor)

        return torch.rand((1,), requires_grad=True)


def launch_thundeRL(
    args: List[str], training_step_mock, input_dir, dataset_dir, output_dir
):
    # split arguments that contain spaces
    args.append(f"--data_layout.input_data.pddl_domains {input_dir}")
    args.append(f"--data_layout.input_data.dataset_dir {dataset_dir}")
    args.append(f"--data_layout.output_data.out_dir {output_dir}")
    args: List[str] = list(
        itertools.chain.from_iterable([arg.split(" ") for arg in args])
    )

    # populate sys.argv with the args
    sys.argv = ["run_lightning_fast.py"] + args

    mockito.patch(
        rgnet.rl.thundeRL.policy_gradient.lit_module.PolicyGradientLitModule,
        "training_step",
        training_step_mock,
    )
    cli = cli_main()
    mockito.unstub(rgnet.rl.thundeRL.policy_gradient.lit_module.PolicyGradientLitModule)
    return cli


def _create_data_setup(tmp_path):
    project_root = Path(__file__).parent.parent
    if project_root.name == "rgnet":
        project_root = project_root / "test"

    data_dir = project_root / "pddl_instances" / "blocks"
    input_dir = tmp_path / "input" / "pddl_domains"
    blocks_dir = input_dir / "blocks"
    # Copy files from data_dir to input_dir
    train_dir = blocks_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    # copy files from data_dir to input_dir except domain.pddl
    for file in [data_dir / "small.pddl", data_dir / "medium.pddl"]:
        shutil.copy(file, train_dir / file.name)
    shutil.copy(data_dir / "domain.pddl", blocks_dir)

    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    dataset_dir = input_dir.parent / "flash_drives" / "blocks"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, out_dir, dataset_dir


def validate_hdata(hetero_data: HeteroData, drives: List[FlashDrive]):
    idx = hetero_data.idx.item()
    return any(
        len(drive) > idx and hetero_data_equal(hetero_data, drive[idx])
        for drive in drives
    )


def validate_batch(hetero_batch: Batch, drives: List[FlashDrive]):
    batch_list = hetero_batch.to_data_list()
    hetero_data: HeteroData
    for hetero_data in batch_list:
        assert validate_hdata(hetero_data, drives)


def validate_successor_batch(
    batch_tuple: Tuple[Batch, Batch, torch.Tensor], drives: List[FlashDrive]
):
    hetero_batch, successor_batch, num_successors = batch_tuple
    # The flattened list of successors
    successor_list: List = successor_batch.to_data_list()
    num_successors_list = num_successors.tolist()
    state_indices_list = hetero_batch.idx.tolist()
    state_succ_offset = 0
    for state_index, num_targets in zip(state_indices_list, num_successors_list):
        found_match = False
        for drive in drives:
            if len(drive) <= state_index:
                continue
            state_hetero_data = drive[state_index]
            targets = state_hetero_data.targets
            if len(targets) != num_targets:
                continue
            # successors are the successors of this state(_index)
            successors = successor_list[
                state_succ_offset : state_succ_offset + num_targets
            ]
            if not all(hetero_data_equal(t, s) for t, s in zip(targets, successors)):
                continue
            found_match = True
        state_succ_offset += num_targets
        if not found_match:
            pytest.fail(
                f"Could not find a state in any drive that matched the index {state_index} in targets."
            )


def _validate_done_reward_num_transitions(
    small_space: ExpandedStateSpaceEnv,
    medium_space: ExpandedStateSpaceEnv,
    mock: PolicyGradientLitModuleMock,
):
    assert len(mock.batched_list) == 5
    # we assert that every state of both problems was encountered once
    flattened_indices = list(
        itertools.chain.from_iterable(
            [batch.idx.tolist() for batch in mock.batched_list]
        )
    )
    # small has 5 states those indices appear in both spaces
    assert flattened_indices.count(0) == 2
    assert flattened_indices.count(1) == 2
    assert flattened_indices.count(2) == 2
    assert flattened_indices.count(3) == 2
    assert flattened_indices.count(4) == 2
    assert len(set(flattened_indices)) == 125

    # Verify done information
    def get_goal_transitions(env):
        return sum(
            len(env.get_applicable_transitions([s])[0])
            for s in env.active_instances[0].goal_states_iter()
        )

    goal_transitions = get_goal_transitions(small_space) + get_goal_transitions(
        medium_space
    )
    flattened_done = torch.cat([batch.done for batch in mock.batched_list])
    assert flattened_done.count_nonzero() == goal_transitions
    done_indices = flattened_done.nonzero().squeeze()
    flattened_reward = torch.cat([batch.reward for batch in mock.batched_list])
    # assert that the rewards are 0 for all done_indices
    assert flattened_reward[done_indices].sum() == 0
    # assert that the rewards are 1 for all other indices
    all_transitions = (
        small_space.active_instances[0].total_transition_count
        + medium_space.active_instances[0].total_transition_count
        - sum(
            small_space.active_instances[0].forward_transition_count(s)
            for s in small_space.active_instances[0].goal_states_iter()
        )
        - sum(
            medium_space.active_instances[0].forward_transition_count(s)
            for s in medium_space.active_instances[0].goal_states_iter()
        )
    )
    print(
        all_transitions,
        flattened_reward[~flattened_done].sum(),
        flattened_reward[~flattened_done].shape,
    )
    assert torch.allclose(
        flattened_reward[~flattened_done].sum().abs(),
        torch.tensor(
            all_transitions,
            dtype=torch.float,
            device=flattened_reward.device,
        ),
    )


@pytest.mark.parametrize("width", [0, 1, 2])
def test_full_epoch_data_collection(tmp_path, width):
    """Run a full epoch of the training setup including small and medium blocks.
    Test that the model receives the correct data by patching it with PolicyGradientModuleMock
    which simply records all incoming data.
    This test might take a bit longer, you can exclude it by adding `--ignore=test/integration` to your pytest script
    """
    input_dir, output_dir, dataset_dir = _create_data_setup(tmp_path)
    mock = PolicyGradientLitModuleMock()
    # # args are specified in config.yaml
    config_files = [
        Path(__file__).parent / f"config.yaml",
        (Path(__file__).parent / f"config{f'-w{width}'}.yaml") if width > 0 else "",
    ]
    cli = launch_thundeRL(
        ["fit"]
        + [f"--config {config_file}" for config_file in config_files if config_file],
        mock.training_step_mock,
        input_dir,
        dataset_dir,
        output_dir,
    )
    data = cli.datamodule
    drives = data.dataset.datasets
    _validate_done_reward_num_transitions(
        data.envs[drives[0].problem_path],
        data.envs[drives[1].problem_path],
        mock,
    )
    for batch_tuple in zip(
        mock.batched_list, mock.successor_batch_list, mock.num_successor_list
    ):
        validate_batch(batch_tuple[0], drives)
        validate_successor_batch(batch_tuple, drives)
