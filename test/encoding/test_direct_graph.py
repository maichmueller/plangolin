from test.fixtures import direct_encoded_state

import pytest

from rgnet.encoding import DirectGraphEncoder
from rgnet.encoding.node_names import node_of


@pytest.mark.parametrize(
    "direct_encoded_state",
    [["blocks", "small", "initial"]],
    indirect=True,
)
def test_direct_encoding_initial(direct_encoded_state):
    graph, _ = direct_encoded_state
    # 2 objects, 1 auxiliary obj
    assert len(graph.nodes) == 2 + 1
    assert (
        str(DirectGraphEncoder._auxiliary_node) in graph.nodes
        and "a" in graph.nodes
        and "b" in graph.nodes
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())
    assert graph.in_degree(node_of(DirectGraphEncoder._auxiliary_node)) == 0


@pytest.mark.parametrize(
    "direct_encoded_state",
    [["blocks", "small", "goal"]],
    indirect=True,
)
def test_direct_encoding_goal(direct_encoded_state):
    graph, _ = direct_encoded_state
    # 2 objects, 1 auxiliary obj
    assert 2 + 1 == len(graph.nodes)
    assert (
        str(DirectGraphEncoder._auxiliary_node) in graph.nodes
        and "a" in graph.nodes
        and "b" in graph.nodes
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())
    assert graph.in_degree(node_of(DirectGraphEncoder._auxiliary_node)) == 0


@pytest.mark.parametrize(
    "direct_encoded_state",
    [["blocks", "large", "goal"]],
    indirect=True,
)
def test_direct_encoding_large_goal(direct_encoded_state):
    graph, _ = direct_encoded_state
    # 7 objects, 1 auxiliary obj
    assert 7 + 1 == len(graph.nodes)
    assert str(DirectGraphEncoder._auxiliary_node) in graph.nodes and all(
        node in graph.nodes for node in "abcdefg"
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())
    assert graph.in_degree(node_of(DirectGraphEncoder._auxiliary_node)) == 0
