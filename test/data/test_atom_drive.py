from test.fixtures import *  # noqa: F401, F403
from test.supervised.test_data import hetero_data_equal

import mockito
from torch_geometric.data import HeteroData

from rgnet.encoding import HeteroGraphEncoder
from rgnet.logging_setup import tqdm
from xmimir.iw import CollectorHook, IWSearch


def validate_drive(drive: AtomDrive, space):
    assert len(drive) == len(space)
    encoder = HeteroGraphEncoder(space.problem.domain)
    iw_search = IWSearch(2)
    for state in tqdm(space, desc="Validating AtomDrive"):
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
        expected = encoder.to_pyg_data(encoder.encode(state))
        assert hetero_data_equal(data, expected)
        # litmus test: any atom in blocksworld/delivery is at most of width 2, so IW(2) should find us all the distances
        # of the atoms from any state
        collector = CollectorHook()
        iw_search.solve(
            space.successor_generator, start_state=state, novel_hook=collector
        )
        atom_dists = data.atom_distances
        for node in collector.nodes:
            for atom_tuples in node.novelty_trace[-1]:
                if len(atom_tuples) == 1:
                    for atom in atom_tuples:
                        atom_str = str(atom)
                        assert atom_str in atom_dists
                        dist = atom_dists[atom_str]
                        assert (atom_str, dist) == (atom_str, node.depth)


@pytest.mark.parametrize(
    ("problem", "fresh_atomdrive"),
    [
        ("small_blocks", ["blocks", "small.pddl"]),
        ("medium_blocks", ["blocks", "medium.pddl"]),
        ("small_delivery_1_pkgs", ["delivery", "instance_2x2_p-1_0.pddl"]),
        ("small_delivery_2_pkgs", ["delivery", "instance_2x2_p-2_0.pddl"]),
    ],
    indirect=["fresh_atomdrive"],  # only this one is a fixture
)
def test_process(problem, fresh_atomdrive, request):
    # Dynamically get the fixture by name
    problem = request.getfixturevalue(problem)
    validate_drive(fresh_atomdrive, problem[0])


@pytest.mark.parametrize(
    ("problem", "fresh_atomdrive"),
    [
        ("small_blocks", ["blocks", "small.pddl"]),
        ("medium_blocks", ["blocks", "medium.pddl"]),
    ],
    indirect=["fresh_atomdrive"],  # only this one is a fixture
)
def test_save_and_load(problem, fresh_atomdrive, request):
    problem = request.getfixturevalue(problem)
    mockito.spy2(AtomDrive.process)
    domain = str(
        fresh_atomdrive.domain_path.parent.relative_to(
            fresh_atomdrive.domain_path.parent.parent
        )
    )
    drive = make_fresh_atomdrive(
        Path(fresh_atomdrive.root).parent,
        domain=domain,
        problem=f"{fresh_atomdrive.problem_path.stem}.pddl",
        force_reload=False,
    )

    mockito.verify(AtomDrive, times=0).process()
    mockito.unstub(AtomDrive)

    validate_drive(drive, problem[0])
