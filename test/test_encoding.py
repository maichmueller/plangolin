import matplotlib.pyplot as plt
import networkx as nx
import pymimir as mi
import pytest

from rgnet.encoding import ColorGraphEncoder, DirectStateEncoder


def _draw_networkx_graph(graph: nx.Graph, **kwargs):
    nx.draw_networkx(
        graph,
        with_labels=kwargs.get("with_labels", True),
        labels=kwargs.get("labels", {n: str(n) for n in graph.nodes}),
        nodelist=kwargs.get("nodelist", [n for n in graph.nodes]),
        node_color=kwargs.get(
            "node_color", [attr["feature"] for _, attr in graph.nodes.data()]
        ),
        cmap=kwargs.get("cmap", "tab10"),
        **kwargs,
    )
    plt.show()


@pytest.fixture
def minimal_blocks_setup():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/minimal.pddl").parse(domain)
    return (
        mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem)),
        domain,
        problem,
    )


@pytest.fixture
def color_encoded_initial_state(minimal_blocks_setup, request):
    space, domain, _ = minimal_blocks_setup
    initial_state = space.get_initial_state()
    return (
        ColorGraphEncoder(domain, add_global_predicate_nodes=request.param).encode(
            initial_state
        ),
        request.param,
    )


@pytest.fixture
def color_encoded_goal_state(minimal_blocks_setup, request):
    space, domain, _ = minimal_blocks_setup
    initial_state = space.get_goal_states()[0]
    return (
        ColorGraphEncoder(domain, add_global_predicate_nodes=request.param).encode(
            initial_state
        ),
        request.param,
    )


@pytest.fixture
def direct_encoded_initial_state(minimal_blocks_setup):
    space, domain, _ = minimal_blocks_setup
    state = space.get_initial_state()
    return DirectStateEncoder(domain).encode(state)


@pytest.fixture
def direct_encoded_goal_state(minimal_blocks_setup):
    space, domain, _ = minimal_blocks_setup
    state = space.get_goal_states()[0]
    return DirectStateEncoder(domain).encode(state)


@pytest.mark.parametrize("color_encoded_initial_state", [True, False], indirect=True)
def test_color_encoding_initial(color_encoded_initial_state):
    graph, with_global_preds = color_encoded_initial_state
    if with_global_preds:
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


@pytest.mark.parametrize("color_encoded_goal_state", [False], indirect=True)
def test_color_encoding_goal(color_encoded_goal_state):
    graph, _ = color_encoded_goal_state
    # 2 objects,
    # 1 * clear, 1 * ontable, 2 * on(a,b)_g, 2* on(a,b), 1 * handempty
    assert 2 + (1 + 1 + 2 + 2 + 1) == len(graph.nodes)
    # on is a goal atom and true in the current state
    assert "on(a, b)_g:0" in graph.nodes and "on(a, b):0" in graph.nodes
    # the predicate holding is neither in the state nor in the goal
    assert all("holding" not in node_name for node_name in graph.nodes)
    assert all("feature" in attr for _, attr in graph.nodes.data())


def test_direct_encoding_initial(direct_encoded_initial_state):
    graph = direct_encoded_initial_state
    # 2 objects, 1 auxiliary obj
    assert 2 + 1 == len(graph.nodes)
    assert (
        str(DirectStateEncoder._auxiliary_node) in graph.nodes
        and "a" in graph.nodes
        and "b" in graph.nodes
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())


def test_direct_encoding_goal(direct_encoded_goal_state):
    graph = direct_encoded_goal_state
    # 2 objects, 1 auxiliary obj
    assert 2 + 1 == len(graph.nodes)
    assert (
        str(DirectStateEncoder._auxiliary_node) in graph.nodes
        and "a" in graph.nodes
        and "b" in graph.nodes
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())
