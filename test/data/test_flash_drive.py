from pathlib import Path
from test.fixtures import (  # noqa: F401
    fresh_flashdrive,
    make_fresh_flashdrive,
    medium_blocks,
    small_blocks,
)

import mockito
import pytest
from test_utils import hetero_data_equal
from torch_geometric.data import HeteroData

from rgnet.encoding import HeteroGraphEncoder
from rgnet.rl.data.flash_drive import FlashDrive


def validate_drive(drive, space):
    assert len(drive) == len(space)
    encoder = HeteroGraphEncoder(space.problem.domain)
    for state in space:
        num_transitions = space.forward_transition_count(state)
        i = state.index
        data: HeteroData = drive[i]
        assert data.idx == i
        if space.is_goal(state):
            assert data.done.all()
        else:
            assert not drive.done.all()
            assert data.done.numel() == num_transitions
            assert data.reward.numel() == num_transitions
            assert len(data.targets) == num_transitions
        expected = encoder.to_pyg_data(encoder.encode(state))
        assert hetero_data_equal(data, expected)


@pytest.mark.parametrize(
    ("problem", "fresh_flashdrive"),
    [
        ("small_blocks", ["blocks", "small.pddl"]),
        ("medium_blocks", ["blocks", "medium.pddl"]),
    ],
    indirect=["fresh_flashdrive"],  # only this one is a fixture
)
def test_process(fresh_flashdrive, problem, request):
    # Dynamically get the fixture by name
    problem = request.getfixturevalue(problem)
    validate_drive(fresh_flashdrive, problem[0])


@pytest.mark.parametrize(
    ("problem", "fresh_flashdrive"),
    [
        ("small_blocks", ["blocks", "small.pddl"]),
        ("medium_blocks", ["blocks", "medium.pddl"]),
    ],
    indirect=["fresh_flashdrive"],  # only this one is a fixture
)
def test_save_and_load(fresh_flashdrive, problem, request):
    # Dynamically get the fixture by name
    problem = request.getfixturevalue(problem)
    mockito.spy2(FlashDrive.process)
    domain = str(
        fresh_flashdrive.domain_path.parent.relative_to(
            fresh_flashdrive.domain_path.parent.parent
        )
    )
    drive = make_fresh_flashdrive(
        Path(fresh_flashdrive.root).parent,
        domain=domain,
        problem=f"{fresh_flashdrive.problem_path.stem}.pddl",
        force_reload=False,
    )

    mockito.verify(FlashDrive, times=0).process()
    mockito.unstub(FlashDrive)

    validate_drive(drive, problem[0])
