from test.fixtures import color_encoded_state

import pytest


@pytest.mark.parametrize(
    "color_encoded_state",
    [["blocks", "small", "initial", False], ["blocks", "small", "initial", True]],
    indirect=True,
)
def test_color_encoding_initial(color_encoded_state):
    graph, encoder = color_encoded_state
    if encoder.predicate_nodes_enabled:
        # 2 objects,
        # 2 * clear, 2 * ontable, 2 * on(a,b)_g (pos 0/1), 1 * handempty,
        # 2 global predicate nodes (clear, ontable)
        # 2 global goal literal nodes (clear_g, ontable_g)
        assert len(graph.nodes) == 2 + (2 + 2 + 2 + 1) + 2 + 2
        assert all("feature" in attr for _, attr in graph.nodes.data())
    else:
        # 2 objects,
        # 2 * clear, 2 * ontable, 2 * on(a,b)_g (pos 0/1), 1 * handempty
        assert len(graph.nodes) == 2 + (2 + 2 + 2 + 1)
        # the predicate holding is neither in the state nor in the goal
        assert all("holding" not in node_name for node_name in graph.nodes)
        assert all("feature" in attr for _, attr in graph.nodes.data())


@pytest.mark.parametrize(
    "color_encoded_state",
    [["blocks", "small", "goal", False]],
    indirect=True,
)
def test_color_encoding_goal(color_encoded_state):
    graph, _ = color_encoded_state
    # 2 objects,
    # 1 * clear, 1 * ontable, 2 * on(a,b)_g, 2* on(a,b), 1 * handempty
    assert len(graph.nodes) == 2 + (1 + 1 + 2 + 2 + 1)
    # on is a goal atom and true in the current state
    assert "(on a b)_g:0" in graph.nodes and "(on a b):0" in graph.nodes
    # the predicate holding is neither in the state nor in the goal
    assert all("holding" not in node_name for node_name in graph.nodes)
    assert all("feature" in attr for _, attr in graph.nodes.data())
