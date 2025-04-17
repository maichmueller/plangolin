from pathlib import Path
from test.fixtures import fresh_drive, make_fresh_drive, medium_blocks  # noqa: F401
from test.supervised.test_data import hetero_data_equal

import mockito
from torch_geometric.data import HeteroData

from rgnet.encoding import HeteroGraphEncoder
from rgnet.rl.thundeRL.flash_drive import FlashDrive


def validate_drive(drive, space):
    assert len(drive) == len(space)
    encoder = HeteroGraphEncoder(space.problem.domain)
    for state in space:
        num_transitions = space.forward_transition_count(state)
        i = state.index
        data: HeteroData = drive[i]
        assert data.idx == i
        assert data.done.numel() == num_transitions
        assert data.reward.numel() == num_transitions
        assert len(data.targets) == num_transitions
        if space.is_goal(state):
            assert data.done.all()
        else:
            assert not drive.done.all()
        expected = encoder.to_pyg_data(encoder.encode(state))
        assert hetero_data_equal(data, expected)


def test_process(fresh_drive, medium_blocks):
    validate_drive(fresh_drive, medium_blocks[0])


def test_save_and_load(fresh_drive, medium_blocks):
    mockito.spy2(FlashDrive.process)

    drive = make_fresh_drive(Path(fresh_drive.root).parent, force_reload=False)

    mockito.verify(FlashDrive, times=0).process()

    validate_drive(drive, medium_blocks[0])
