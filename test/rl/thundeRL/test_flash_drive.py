from pathlib import Path
from test.fixtures import fresh_drive, medium_blocks

import mockito
from supervised.test_data import assert_hetero_stores
from torch_geometric.data import HeteroData

from rgnet.encoding import HeteroGraphEncoder
from rgnet.encoding.base_encoder import EncoderFactory
from rgnet.rl.thundeRL.flash_drive import FlashDrive


def validate_drive(drive, space):
    assert len(drive) == space.num_states()
    encoder = HeteroGraphEncoder(space.problem.domain)
    for i, state in enumerate(space.get_states()):
        num_transitions = len(space.get_forward_transitions(state))
        data: HeteroData = drive[i]
        assert data.idx == i
        assert data.done.numel() == num_transitions
        assert data.reward.numel() == num_transitions
        assert len(data.targets) == num_transitions
        if space.is_goal_state(state):
            assert data.done.all()
        else:
            assert not drive.done.all()
        expected = encoder.to_pyg_data(encoder.encode(state))
        assert_hetero_stores(data, expected)


def test_process(fresh_drive, medium_blocks):
    validate_drive(fresh_drive, medium_blocks[0])


def test_save_and_load(fresh_drive, medium_blocks):
    data_dir = Path(__file__).parent.parent.parent / "pddl_instances" / "blocks"
    problem_path = data_dir / "medium.pddl"
    domain_path = data_dir / "domain.pddl"

    mockito.spy2(FlashDrive.process)

    drive = FlashDrive(
        problem_path=problem_path,
        domain_path=domain_path,
        custom_dead_end_reward=-100.0,
        root_dir=fresh_drive.root,
        force_reload=False,
        encoder_factory=EncoderFactory(HeteroGraphEncoder),
    )

    mockito.verify(FlashDrive, times=0).process()

    validate_drive(drive, medium_blocks[0])
