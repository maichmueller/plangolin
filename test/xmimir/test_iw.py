from test.fixtures import *  # noqa: F403,F401

import pytest

from xmimir.iw import *


@pytest.mark.parametrize(
    "space_fixture",
    [
        "small_blocks",
        "medium_blocks",
        "medium_blocks_width1_goal",
        "largish_blocks_width2_goal",
        "largish_blocks_unbound_goal",
    ],
)
@pytest.mark.parametrize("width", [1, 2])
def test_iw_search(space_fixture, width, request):
    space, domain, problem = request.getfixturevalue(space_fixture)
    # these are the problem names in the pddl files of
    # small and medium blocks problems. This entire test is hardcoded to these files, so if they change this
    # test may need adaptation!!
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
            decomposition = [[str(next(problem.goal()).atom)]]
        case _:
            raise ValueError(f"Problem {problem.name} not tested.")

    iw = IWSearch(width, expansion_strategy=InOrderExpansion())
    prev_state = space.initial_state
    state = prev_state
    # for each subgoal: run iw from the current state and check if the subgoal is in the novelty traces
    # of the found novel nodes. If so, jump to that state and repeat.
    for subgoals in decomposition:
        print("Current state", state)
        print("Subgoals:    ", subgoals)
        collector = CollectorHook()
        iw.solve(
            space.successor_generator,
            start_state=state,
            stop_on_goal=False,
            novel_hook=collector,
        )
        viable_nodes = []
        for node in collector.nodes:
            # expectation: atoms represent themselves as a string of the form: "(predicate obj_1 obj_2 ... obj_N)"
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

    assert any(state.semantic_eq(state) for state in space.goal_states_iter())
