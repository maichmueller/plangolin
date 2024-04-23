import matplotlib.pyplot as plt
import networkx as nx
import pymimir as mi
import pytest

from rgnet.encoding import ColorGraphEncoder, DirectStateEncoder
from rgnet.encoding.node_names import node_of


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


def problem_setup(domain_name, problem):
    domain = mi.DomainParser(f"test/pddl_instances/{domain_name}/domain.pddl").parse()
    problem = mi.ProblemParser(
        f"test/pddl_instances/{domain_name}/{problem}.pddl"
    ).parse(domain)
    return (
        mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem)),
        domain,
        problem,
    )


@pytest.fixture
def color_encoded_state(request):
    domain_param, prob_param, which_state_param, add_param = request.param
    space, domain, _ = problem_setup(domain_param, prob_param)
    if which_state_param == "initial":
        state = space.get_initial_state()
    elif which_state_param == "goal":
        state = space.get_goal_states()[0]
    return (
        ColorGraphEncoder(domain, add_global_predicate_nodes=add_param).encode(state),
        add_param,
    )


@pytest.fixture
def direct_encoded_initial_state(request):
    domain_param, prob_param, which_state_param, add_param = request.param
    space, domain, _ = problem_setup(domain_param, prob_param)
    if which_state_param == "initial":
        state = space.get_initial_state()
    else:
        state = space.get_goal_states()[0]
    return (
        DirectStateEncoder(domain).encode(state),
        request.param,
    )


@pytest.mark.parametrize(
    "color_encoded_state",
    [["blocks", "small", "initial", False], ["blocks", "small", "initial", True]],
    indirect=True,
)
def test_color_encoding_initial(color_encoded_state):
    graph, with_global_preds = color_encoded_state
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
    assert "on(a, b)_g:0" in graph.nodes and "on(a, b):0" in graph.nodes
    # the predicate holding is neither in the state nor in the goal
    assert all("holding" not in node_name for node_name in graph.nodes)
    assert all("feature" in attr for _, attr in graph.nodes.data())


@pytest.mark.parametrize(
    "direct_encoded_initial_state",
    [["blocks", "small", "initial", False]],
    indirect=True,
)
def test_direct_encoding_initial(direct_encoded_initial_state):
    graph, _ = direct_encoded_initial_state
    # 2 objects, 1 auxiliary obj
    assert len(graph.nodes) == 2 + 1
    assert (
        str(DirectStateEncoder._auxiliary_node) in graph.nodes
        and "a" in graph.nodes
        and "b" in graph.nodes
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())
    assert graph.in_degree(node_of(DirectStateEncoder._auxiliary_node)) == 0


@pytest.mark.parametrize(
    "direct_encoded_initial_state",
    [["blocks", "small", "goal", False]],
    indirect=True,
)
def test_direct_encoding_goal(direct_encoded_initial_state):
    graph, _ = direct_encoded_initial_state
    # 2 objects, 1 auxiliary obj
    assert 2 + 1 == len(graph.nodes)
    assert (
        str(DirectStateEncoder._auxiliary_node) in graph.nodes
        and "a" in graph.nodes
        and "b" in graph.nodes
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())
    assert graph.in_degree(node_of(DirectStateEncoder._auxiliary_node)) == 0


@pytest.mark.parametrize(
    "direct_encoded_initial_state",
    [["blocks", "large", "goal", False]],
    indirect=True,
)
def test_direct_encoding_large_goal(direct_encoded_initial_state):
    graph, _ = direct_encoded_initial_state
    # 7 objects, 1 auxiliary obj
    assert 7 + 1 == len(graph.nodes)
    assert str(DirectStateEncoder._auxiliary_node) in graph.nodes and all(
        node in graph.nodes for node in "abcdefg"
    )
    assert all("feature" in attr for *rest, attr in graph.edges.data())
    assert graph.in_degree(node_of(DirectStateEncoder._auxiliary_node)) == 0
