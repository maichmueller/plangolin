import operator
from test.fixtures import *  # noqa: F401, F403
from test.supervised.test_data import hetero_data_equal
from typing import Callable

import mockito
from torch_geometric.data import HeteroData

from rgnet.algorithms import mdp_graph_as_pyg_data
from rgnet.algorithms.policy_evaluation_mp import OptimalAtomValuesMP
from rgnet.encoding import HeteroGraphEncoder
from rgnet.logging_setup import tqdm
from rgnet.rl.data.atom_drive import make_atom_ids
from xmimir import XAtom
from xmimir.iw import CollectorHook, IWSearch


def validate_drive(drive: AtomDrive, space, invert_values: bool = False):
    assert len(drive) == len(space)
    encoder = HeteroGraphEncoder(space.problem.domain)
    sign = -1 if invert_values else 1
    expected_values = drive.atom_value_dict_to_tensor(
        dict(
            sorted(
                _expected_optimal_atom_values(space).items(),
                key=operator.itemgetter(0),
            )
        )
    )
    computed_values = drive.atom_values.view(len(drive), -1)
    assert computed_values.shape == expected_values.shape
    both_finite = ~(torch.isinf(computed_values) & torch.isinf(expected_values))
    if not torch.allclose(
        sign * computed_values[both_finite], expected_values[both_finite]
    ):
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
            atom_values = data.atom_values
            validate_atom_values(
                lambda atom_str: (
                    sign * atom_values[drive.atom_to_index_map[atom_str]]
                ).item(),
                space,
                state,
            )


def validate_atom_values(atom_dists: Callable[[str], float], space, state):
    expected_values = _expected_optimal_atom_values(space, state)[state.index]
    computed = {atom: atom_dists(str(atom)) for atom in expected_values.keys()}
    state_str = str(state)
    assert (state_str, {str(a): v for a, v in computed.items()}) == (
        state_str,
        {str(a): v for a, v in expected_values.items()},
    )


def _expected_optimal_atom_values(space, state=None) -> dict[int, dict[XAtom, float]]:
    # litmus test: any atom in blocksworld/delivery is at most of width 2, so IW(2) should find us all the distances
    # of the atoms from any state
    if state is None:
        states = space
    else:
        states = [state]
    expected_values = {i: dict() for i in range(len(states))}
    for state in states:
        collector = CollectorHook()
        iw_search = IWSearch(2, depth_1_is_novel=False)
        iw_search.solve(
            space.successor_generator,
            start_state=state,
            novel_hook=collector,
            stop_on_goal=False,
        )
        expected_values[state.index] = {
            atom: 0.0 for atom in state.atoms(with_statics=False)
        }
        for node in collector.nodes:
            for atom_tuples in node.novelty_trace[-1]:
                for atom in atom_tuples:
                    if len(atom_tuples) == 1:
                        expected_values[state.index][atom] = float(node.depth)
    return expected_values


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
    validate_drive(fresh_atomdrive, problem[0], invert_values=True)


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

    validate_drive(drive, problem[0], invert_values=True)


@pytest.mark.parametrize(
    "problem",
    [
        "small_blocks",
        "medium_blocks",
        "small_delivery_1_pkgs",
        "small_delivery_2_pkgs",
    ],
)
def test_atom_dist_mp_module(request, problem):
    import time

    problem = request.getfixturevalue(problem)
    space = problem[0]
    env = ExpandedStateSpaceEnv(
        space,
        reward_function=UnitReward(gamma=1.0, goal_reward=1.0, regular_reward=1.0),
        batch_size=1,
    )
    start = time.time()
    pyg_env = env.to_pyg_data(0, natural_transitions=True)
    print(f"Time taken for pyg_env conversion: {time.time() - start:.2f} seconds")
    pyg_env.atoms_per_state = [list(state.atoms(with_statics=False)) for state in space]
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    mp_module = OptimalAtomValuesMP(
        atom_to_index_map=make_atom_ids(space.problem)[0], aggr="min"
    ).to(device)
    start = time.time()
    mp_module(pyg_env.to(device))
    print(f"Time taken for MP module: {time.time() - start:.2f} seconds")
    for state in space:
        dist_tensor = getattr(pyg_env, OptimalAtomValuesMP.default_attr_name)[
            state.index
        ]
        validate_atom_values(
            lambda atom_str: dist_tensor[mp_module.atom_to_index[atom_str]].item(),
            space,
            state,
        )


def test_atom_dist_mp_module_manual_graph():
    graph = nx.DiGraph()
    graph.add_node(0, ntype="state")
    graph.add_node(1, ntype="state")
    graph.add_node(2, ntype="state")
    graph.add_node(3, ntype="state")
    graph.add_edge(0, 1, reward=1.0, probs=0.5, done=False, idx=0)
    graph.add_edge(1, 2, reward=1.0, probs=0.5, done=False, idx=1)
    graph.add_edge(2, 0, reward=1.0, probs=0.5, done=False, idx=2)
    graph.add_edge(0, 3, reward=1.0, probs=0.5, done=False, idx=3)

    atom_to_index_map = {
        "t": 0,
        "q": 1,
        "p": 2,
    }
    pyg_graph = mdp_graph_as_pyg_data(graph)
    pyg_graph.atoms_per_state = [["p"], ["q"], ["t"], []]
    mp_module = OptimalAtomValuesMP(
        gamma=1.0, atom_to_index_map=atom_to_index_map, aggr="min"
    )
    mp_module(pyg_graph)
    final_distances = pyg_graph[OptimalAtomValuesMP.default_attr_name]
    expected = torch.tensor(
        [[2, 1, 0], [1, 0, 2], [0, 2, 1], [torch.inf, torch.inf, torch.inf]],
        dtype=torch.float,
    )
    assert final_distances.shape == expected.shape
    for i in range(final_distances.shape[0]):
        for j in range(final_distances.shape[1]):
            if final_distances[i, j] == torch.inf:
                assert expected[i, j] == torch.inf
            else:
                assert final_distances[i, j] == expected[i, j]
    assert torch.allclose(final_distances, expected)
