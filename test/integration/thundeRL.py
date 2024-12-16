import itertools
import shutil
import sys
from pathlib import Path
from test.fixtures import medium_blocks, small_blocks
from typing import Any, List, Tuple

import mockito
import pymimir as mi
import pytest
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch_geometric.data import Batch, HeteroData

import rgnet

# TODO experiments is not a source module
from experiments.rl.thundeRL import run_lightning_fast
from rgnet.encoding import HeteroGraphEncoder
from rgnet.rl.thundeRL.flash_drive import FlashDrive

from ..supervised.test_data import hetero_data_equal


@pytest.fixture(autouse=True, scope="class")
def setup_multiprocessing():
    torch.multiprocessing.set_start_method("fork", force=True)


class PolicyGradientModuleMock:

    def __init__(self):
        super().__init__()
        self.batched_list: List[Batch] = []
        self.successor_batch_list: List[Batch] = []
        self.num_successor_list: List[torch.Tensor] = []

    def training_step_mock(
        self, batch_tuple: Tuple[Batch, Batch, torch.Tensor], **kwargs: Any
    ) -> STEP_OUTPUT:
        assert isinstance(batch_tuple, tuple)
        assert len(batch_tuple) == 3
        assert isinstance(batch_tuple[0], Batch)
        assert isinstance(batch_tuple[1], Batch)
        assert isinstance(batch_tuple[2], torch.Tensor)
        batched, successor_batch, num_successor = batch_tuple
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
        rgnet.rl.thundeRL.lightning_adapter.PolicyGradientModule,
        "training_step",
        training_step_mock,
    )

    run_lightning_fast.cli_main()


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
    if not any(
        len(drive) > idx and hetero_data_equal(hetero_data, drive[idx])
        for drive in drives
    ):
        pytest.fail(str(hetero_data))


def validate_batch(hetero_batch: Batch, drives: List[FlashDrive]):
    batch_list = hetero_batch.to_data_list()
    hetero_data: HeteroData
    for hetero_data in batch_list:
        validate_hdata(hetero_data, drives)


def validate_successor_batch(
    batch_tuple: Tuple[Batch, Batch, torch.Tensor], drives: List[FlashDrive]
):
    hetero_batch, successor_batch, num_successors = batch_tuple
    # The flattened list of successors
    successor_list: List = successor_batch.to_data_list()
    num_successors_list = num_successors.tolist()
    state_indices_list = hetero_batch.idx.tolist()
    successor_pointer = 0
    for state_index, num_targets in zip(state_indices_list, num_successors_list):
        found_match = False
        for drive in drives:
            if len(drive) <= state_index:
                continue
            potential_hdata = drive[state_index]
            targets = potential_hdata.targets
            if len(targets) != num_targets:
                continue
            successors = successor_list[
                successor_pointer : successor_pointer + num_targets
            ]
            if not all(hetero_data_equal(t, s) for t, s in zip(targets, successors)):
                continue
            found_match = True
        successor_pointer += num_targets
        if not found_match:
            pytest.fail(
                "Could not find a state in any drive that matched the index and targets."
                + str(state_index)
            )


def _validate_done_reward_num_transitions(
    small_space: mi.StateSpace,
    medium_space: mi.StateSpace,
    mock: PolicyGradientModuleMock,
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
    def get_goal_transitions(space):
        return sum(
            [len(space.get_forward_transitions(s)) for s in space.get_goal_states()]
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
    all_transitions = small_space.num_transitions() + medium_space.num_transitions()
    assert torch.allclose(
        flattened_reward.sum().abs(),
        torch.tensor([all_transitions - goal_transitions], dtype=torch.float),
    )


def test_full_epoch(tmp_path, small_blocks, medium_blocks):
    """Run a full epoch of the training setup including small and medium blocks.
    Test that the model receives the correct data by patching it with PolicyGradientModuleMock
    which simply records all incoming data.
    """
    small_space: mi.StateSpace
    medium_space: mi.StateSpace
    small_space, domain, small_problem = small_blocks
    medium_space, _, medium_problem = medium_blocks

    input_dir, output_dir, dataset_dir = _create_data_setup(tmp_path)
    domain_path = input_dir / "blocks" / "domain.pddl"
    problem_dir = input_dir / "blocks" / "train"

    drives = [
        FlashDrive(
            domain_path=domain_path,
            problem_path=problem,
            custom_dead_end_reward=-(1.0 / 1.0 - 0.9),
            root_dir=str(dataset_dir),
            encoder_factory=HeteroGraphEncoder,
        )
        for problem in problem_dir.iterdir()
    ]

    assert small_space.num_states() + medium_space.num_states() == 130
    mock = PolicyGradientModuleMock()
    # args are specified in config.yaml
    config_file = Path(__file__).parent / "config.yaml"
    launch_thundeRL(
        ["fit", f"--config {config_file}"],
        mock.training_step_mock,
        input_dir,
        dataset_dir,
        output_dir,
    )

    # _validate_don_reward_num_transitions(small_space, medium_space, mock)

    for batch_tuple in zip(
        mock.batched_list, mock.successor_batch_list, mock.num_successor_list
    ):
        validate_batch(batch_tuple[0], drives)
        validate_successor_batch(batch_tuple, drives)
