from functools import cache

import networkx as nx
import torch
import torch_geometric

from xmimir import StateType


@cache
def mdp_graph_as_pyg_data(nx_state_space_graph: nx.DiGraph):
    """
    Convert the networkx graph into a directed pytorch_geometric graph.
    The reward for each transition is stored in edge_attr[:, 0].
    The transition probabilities are stored in edge_attr[:, 1].
    The node features are stored as usual in graph.x.
    The first dimension is the node value (starting with 0).
    The second node feature dimension is one, if the node is a goal state.
    """
    pyg_graph = torch_geometric.utils.from_networkx(
        nx_state_space_graph, group_edge_attrs=["reward", "probs", "idx"]
    )
    transition_indices = pyg_graph.edge_attr[:, 2]
    expected_transition_indices = torch.arange(transition_indices.max().item() + 1)
    if (transition_indices.int() != expected_transition_indices).any():
        # we have to maintain the order of the edges as they are returned by a traversal of the state space.
        graph_clone = pyg_graph.clone()
        sorted_transition_indices = torch.argsort(graph_clone.edge_attr[:, 2])
        pyg_graph.edge_index = graph_clone.edge_index[:, sorted_transition_indices]
        pyg_graph.edge_attr = graph_clone.edge_attr[sorted_transition_indices, :]
    transition_indices = pyg_graph.edge_attr[:, 2]
    assert (transition_indices.int() == expected_transition_indices).all()
    is_goal_state = [False] * pyg_graph.num_nodes
    # inf as default to trigger errors if logic did not hold
    goal_reward = [float("inf")] * pyg_graph.num_nodes
    for i, (node, attr) in enumerate(nx_state_space_graph.nodes.data()):
        if attr["ntype"] == StateType.GOAL:
            is_goal_state[i] = True
            _, _, goal_reward[i] = next(
                iter(nx_state_space_graph.out_edges(node, data="reward"))
            )

    pyg_graph.goals = torch.tensor(
        is_goal_state,
        dtype=torch.bool,
    )
    # goal states have the value of their reward (typically 0, but could be arbitrary);
    # the rest is initialized to 0.
    pyg_graph.x = torch.where(
        pyg_graph.goals,
        torch.tensor(goal_reward, dtype=torch.float),
        torch.zeros((pyg_graph.num_nodes,)),
    )
    if hasattr(nx_state_space_graph.graph, "gamma"):
        pyg_graph.gamma = nx_state_space_graph.graph["gamma"]
    return pyg_graph
