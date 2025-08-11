from functools import cache

import networkx as nx
import torch
import torch_geometric

from xmimir import StateType


@cache
def mdp_graph_as_pyg_data(nx_state_space_graph: nx.DiGraph):
    """
    Convert a NetworkX state‑space `DiGraph` to a PyTorch Geometric `Data` object.

    Edge attributes (column order as created by `from_networkx(..., group_edge_attrs=["reward", "probs", "done", "idx"])`):
    - `edge_attr[:, 0]`: reward (float)
    - `edge_attr[:, 1]`: transition probability (float)
    - `edge_attr[:, 2]`: terminal flag (bool cast to float)
    - `edge_attr[:, 3]`: traversal index (int). Edges are re-ordered to be increasing in this index to
      preserve the original traversal order.

    Node-level attributes set on the returned graph:
    - `x`: shape `[num_nodes]`, initial value per state. Goal states are initialized to their immediate goal
      reward (often `0.0`), all other states to `0.0`.
    - `goals`: `BoolTensor[num_nodes]`, `True` for goal states.
    - `gamma`: optional float scalar copied from `nx_state_space_graph.graph['gamma']` if present.

    Notes
    -----
    - The result is memoized via `functools.cache` keyed by the input graph object identity.
    - Only the first two edge-attribute columns (reward, probability) are required by the message‑passing
      modules in this package; the others are retained for bookkeeping.

    Returns
    -------
    torch_geometric.data.Data
        The PyG graph with ordered `edge_index/edge_attr`, `x`, `goals`, and optional `gamma`.
    """
    pyg_graph = torch_geometric.utils.from_networkx(
        nx_state_space_graph, group_edge_attrs=["reward", "probs", "done", "idx"]
    )
    trans_attr_idx = 3
    transition_indices = pyg_graph.edge_attr[:, trans_attr_idx]
    expected_transition_indices = torch.arange(transition_indices.max().item() + 1)
    if (transition_indices.int() != expected_transition_indices).any():
        # we have to maintain the order of the edges as they are returned by a traversal of the state space.
        graph_clone = pyg_graph.clone()
        sorted_transition_indices = torch.argsort(
            graph_clone.edge_attr[:, trans_attr_idx]
        )
        pyg_graph.edge_index = graph_clone.edge_index[:, sorted_transition_indices]
        pyg_graph.edge_attr = graph_clone.edge_attr[sorted_transition_indices, :]
    transition_indices = pyg_graph.edge_attr[:, trans_attr_idx]
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
