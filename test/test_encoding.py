import matplotlib.pyplot as plt
import networkx as nx
import pymimir as mi

from rgnet.encoding import ColorGraphEncoder


def _draw_networkx_graph(graph: nx.Graph):
    nx.draw_networkx(
        graph,
        with_labels=True,
        labels={n: str(n) for n in graph.nodes},
        nodelist=[n for n in graph.nodes],
        node_color=[attr["color"] for _, attr in graph.nodes.data()],
        cmap="tab10",
    )
    plt.show()


def test_encoding_initial():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/minimal.pddl").parse(domain)
    state_space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    initial_state = state_space.get_initial_state()
    graph = ColorGraphEncoder(domain).encode(initial_state)
    # 2 objects,
    # 2 * clear, 2 * ontable, 2 * on(a,b)_g, 1 * handempty
    assert 2 + (2 + 2 + 2 + 1) == len(graph.nodes)
    # the predicate holding is neither in the state nor in the goal
    assert all("holding" not in node_name for node_name in graph.nodes)
    assert all("color" in attr for _, attr in graph.nodes.data())


def test_encoding_goal_state():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/minimal.pddl").parse(domain)
    state_space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    goal_state = state_space.get_goal_states()[0]
    graph = ColorGraphEncoder(domain).encode(goal_state)
    # 2 objects,
    # 1 * clear, 1 * ontable, 2 * on(a,b)_g, 2* on(a,b), 1 * handempty
    assert 2 + (1 + 1 + 2 + 2 + 1) == len(graph.nodes)
    # on is a goal atom and true in the current state
    assert "on(a,b)_g:0" in graph.nodes and "on(a,b):0" in graph.nodes
    # the predicate holding is neither in the state nor in the goal
    assert all("holding" not in node_name for node_name in graph.nodes)
    assert all("color" in attr for _, attr in graph.nodes.data())
