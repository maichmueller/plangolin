import matplotlib.pyplot as plt
import networkx as nx
import pymimir as mi

from rgnet.encoding import ColorGraphEncoder
import networkx as nx
import matplotlib.pyplot as plt

draw_graph = True


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
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    state_space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    state = state_space.get_initial_state()
    graph = ColorGraphEncoder(domain).encode(state)
    # 4 objects, 2 predicates of arity one and one goal predicate of arity 2
    assert 4 + (2 + 2) == len(graph.nodes)
    # the predicate holding is neither in the state nor in the goal
    assert all("holding" not in node_name for node_name in graph.nodes)
    assert all("color" in attr for _, attr in graph.nodes.data())


def test_encoding_goal_state():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    state_space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    state = state_space.get_goal_states()[0]
    graph = ColorGraphEncoder(domain).encode(state)
    # 4 objects, (2 +1 * 2) state predicates and one goal predicate of arity 2
    assert 4 + (4 + 2) == len(graph.nodes)
    # on is a goal atom and true in the current state
    assert "p_on_g:0" in graph.nodes and "p_on:0" in graph.nodes
    # the predicate holding is neither in the state nor in the goal
    assert all("holding" not in node_name for node_name in graph.nodes)
    assert all("color" in attr for _, attr in graph.nodes.data())
