import pickle
from test.fixtures import *  # noqa: F403,F401
from typing import Sequence

from xmimir import XState, XSuccessorGenerator
from xmimir.iw import *


@pytest.mark.parametrize(
    "problem",
    [
        "small",
        "medium",
        "iw/medium_width1_goal",
        "iw/largish_width2_goal",
        "iw/largish_unbound_goal",
    ],
)
@pytest.mark.parametrize("width", [1, 2])
def test_siw(problem, width):
    source_dir = "" if os.getcwd().endswith("/test") else "test/"
    domain_path = f"{source_dir}pddl_instances/blocks/domain.pddl"
    problem_path = f"{source_dir}pddl_instances/blocks/{problem}.pddl"
    domain, problem = xmi.parse(domain_path, problem_path)
    assert problem.domain.name == "blocks"
    # these are the problem names in the pddl files of
    # small and medium blocks problems. This entire test is hardcoded to these files, so if they change this
    # test may need adaptation!!
    # expectation: atoms represent themselves as a string of the form: "(predicate obj_1 obj_2 ... obj_N)"
    match problem.name:
        case "blocks-2-0":
            decomposition = [["(on a b)"]]
        case "blocks-4-0":
            decomposition = [["(on b a)"], ["(on c b)"], ["(on d c)"]]
        case "blocks-6-w404":
            match width:
                case 1:
                    decomposition = [
                        # first we need to have all blocks on the table, we cannot simply do 'ontable b' because iw-1
                        # may place the upper blocks anywhere
                        ["(ontable c)"],
                        ["(ontable d)"],
                        ["(ontable b)"],
                        ["(ontable a)"],
                        ["(on c d)"],
                        ["(on f c)"],
                        ["(on b f)"],
                        ["(on e b)"],
                        ["(on a e)"],
                    ]
                case 2:
                    decomposition = [
                        ["(ontable d)"],
                        ["(on c d)"],
                        ["(on f c)"],
                        ["(on b f)"],
                        ["(on e b)"],
                        ["(on a e)"],
                    ]
                case _:
                    raise ValueError(f"Width {width} not tested.")
        case "blocks-6-w2":
            match width:
                case 1:
                    decomposition = [
                        ["(ontable a)"],
                        ["(ontable b)"],
                        ["(ontable d)"],
                        ["(on e f)"],
                    ]
                case 2:
                    decomposition = [["(on e f)"]]
                case _:
                    raise ValueError(f"Width {width} not tested.")
        case "blocks-5-w1":
            decomposition = [[str(tuple(problem.goal())[0].atom)]]
        case _:
            raise ValueError(f"Problem {problem.name} not tested.")

    successor_gen = xmi.XSuccessorGenerator(problem)
    state = siw(successor_gen.initial_state, decomposition, successor_gen, width)

    assert state.is_goal


def siw(
    init_state: XState,
    decomposition: Sequence[Sequence[str]],
    successor_generator: XSuccessorGenerator,
    width: int,
):
    iw = IWSearch(width, expansion_strategy=InOrderExpansion())
    prev_state = init_state
    state = prev_state
    # for each subgoal: run iw from the current state and check if the subgoal is in the novelty traces
    # of the found novel nodes. If so, jump to that state and repeat.
    for subgoals in decomposition:
        print("Current state", state)
        print("Subgoals:    ", subgoals)
        collector = CollectorHook()
        iw.solve(
            successor_generator,
            start_state=state,
            stop_on_goal=False,
            novel_hook=collector,
        )
        viable_nodes = []
        for node in collector.nodes:
            if all(
                any(
                    str(atom) == subgoal
                    for atom_tuples in node.novelty_trace
                    for atom_tuple in atom_tuples
                    for atom in atom_tuple
                )
                for subgoal in subgoals
            ):
                viable_nodes.append(node)
        if not viable_nodes:
            assert (
                False
            ), f"Could not find subgoal {subgoals} in the novelty traces of found novel nodes from state={str(state)}"
        optimal_node = min(viable_nodes, key=lambda node: node.depth)
        prev_state = state
        state = optimal_node.state
        assert prev_state != state
    return state


@pytest.mark.parametrize(
    "space_fixture, expected_solution_upper_bound_cost",
    [
        ("small_blocks", 1),
        ("medium_blocks", 3),
        ("largish_blocks_unbound_goal", 9),
    ],
)
def test_iw1_state_space(space_fixture, expected_solution_upper_bound_cost, request):
    space, domain, problem = request.getfixturevalue(space_fixture)
    iw_space = IWStateSpace(IWSearch(1), space, n_cpus=os.cpu_count(), chunk_size=100)
    assert (
        0
        < iw_space.goal_distance(iw_space.initial_state)
        <= expected_solution_upper_bound_cost
    )


@pytest.mark.parametrize(
    "space_fixture",
    [
        "small_blocks",
        "medium_blocks",
    ],
)
def test_iw1_state_serialization(space_fixture, request):
    space, domain, problem = request.getfixturevalue(space_fixture)
    iw_search = IWSearch(1)
    pickled_obj = pickle.dumps(iw_search)
    iw = pickle.loads(pickled_obj)
    assert isinstance(iw, IWSearch)
    assert iw.width == 1
    assert iw.expansion_strategy == iw_search.expansion_strategy
    iw_space = IWStateSpace(iw_search, space, n_cpus=os.cpu_count(), chunk_size=300)
    pickled_space = pickle.dumps(iw_space)
    deserialized_space = pickle.loads(pickled_space)
    assert isinstance(deserialized_space, IWStateSpace)

    for state in iw_space:
        assert deserialized_space[state.index].semantic_eq(state)
        assert state.index == deserialized_space[state.index].index
        assert iw_space.goal_distance(state) == deserialized_space.goal_distance(
            deserialized_space[state.index]
        )


#
# @pytest.mark.parametrize(
#     "space_fixture, expected_solution_upper_bound_cost",
#     [
#         ("small_blocks", 1),
#         ("medium_blocks", 3),
#         ("largish_blocks_unbound_goal", 6),
#     ],
# )
# def test_iw2_state_space(space_fixture, expected_solution_upper_bound_cost, request):
#     space, domain, problem = request.getfixturevalue(space_fixture)
#     iw_space = IWStateSpace(IWSearch(2), space)
#     assert iw_space.goal_distance(iw_space.initial_state) == expected_solution_upper_bound_cost
