import dataclasses
import pathlib
from typing import Dict, Set, Tuple

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import HeteroData

from plangolin.utils import ftime, import_all_from
from plangolin.utils.misc import broadcastable
from xmimir import XDomain, XProblem


def test_ftime():
    assert "100us" == ftime(0.0001)
    assert "100ms" == ftime(0.1)
    assert "14s" == ftime(14)
    assert "01:00m" == ftime(60)
    assert "01:01m" == ftime(61)
    assert "1:00:00h" == ftime(3600)
    assert "1:00:01h" == ftime(3601)
    assert "01:00m" == ftime(60.1)


def test_import_all_from():
    path = "test/pddl_instances/blocks"
    domain, problems = import_all_from(
        "test/pddl_instances/blocks", domain_filename="domain"
    )
    assert isinstance(domain, XDomain)
    assert all(
        isinstance(prob, XProblem) and prob.domain.filepath == domain.filepath
        for prob in problems
    )
    # -1 as one pddl file is the domain
    assert len(problems) == len(list(pathlib.Path(path).glob("*.pddl"))) - 1


@dataclasses.dataclass
class GridWorldInstance:
    """
    Configuration for a deterministic GridWorld MDP:
      - n_rows, n_cols: grid dimensions
      - obstacles: set of (i,j) coordinates to block
      - terminals: mapping from (i,j) to terminal reward
      - step_reward: reward per non-terminal step
    """

    n_rows: int
    n_cols: int
    obstacles: Set[Tuple[int, int]]
    terminals: Dict[Tuple[int, int], float]
    step_reward: float = -0.04
    graph: nx.DiGraph = dataclasses.field(init=False)

    def __post_init__(self):
        self.graph = make_gridworld(
            self.n_rows,
            self.n_cols,
            self.obstacles,
            self.terminals,
            self.step_reward,
        )


def make_gridworld(
    n_rows: int,
    n_cols: int,
    obstacles: Set[Tuple[int, int]],
    terminals: Dict[Tuple[int, int], float],
    step_reward: float = -0.04,
) -> nx.DiGraph:
    """
    Constructs a deterministic GridWorld MDP as a NetworkX DiGraph with:
      - nodes: (i, j) cells
      - edges: deterministic transitions with attributes 'action', 'prob'=1.0, 'reward'
    """
    G = nx.DiGraph()
    actions = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

    def in_bounds(i: int, j: int) -> bool:
        return 0 <= i < n_rows and 0 <= j < n_cols

    # add all non-obstacle cells
    for i in range(n_rows):
        for j in range(n_cols):
            if (i, j) not in obstacles:
                G.add_node((i, j))

    for s in list(G.nodes):
        # terminal: self-loop only
        if s in terminals:
            G.add_edge(s, s, action="TERMINAL", prob=1.0, reward=terminals[s])
            continue
        # deterministic transitions
        for a, (di, dj) in actions.items():
            ni, nj = s[0] + di, s[1] + dj
            # move or stay if blocked
            if not in_bounds(ni, nj) or (ni, nj) in obstacles:
                sp = s
            else:
                sp = (ni, nj)
            G.add_edge(s, sp, action=a, prob=1.0, reward=step_reward)
    return G


def _debug_plotit(graph, goal_state):
    G_multi = nx.MultiDiGraph()
    num_edges = graph.edge_index.size(1)
    for i in range(num_edges):
        u = int(graph.edge_index[0, i].item())
        v = int(graph.edge_index[1, i].item())
        prob = graph.edge_attr[i, 0].item()  # transition probability
        reward = graph.edge_attr[i, 1].item()  # reward
        G_multi.add_edge(u, v, prob=prob, reward=reward)

    # Compute a layout.
    pos = nx.spring_layout(G_multi, seed=4)

    # Define node colors: red for the goal node, blue for others.
    node_colors = [
        "red" if node == goal_state.index else "blue" for node in G_multi.nodes()
    ]

    # ------------------------------------------------
    # Compute curvature for bidirectional (or multiedge) pairs
    # ------------------------------------------------
    # Group edges by unordered node pairs.
    edge_groups = {}
    for u, v, key in G_multi.edges(keys=True):
        key_pair = tuple(sorted((u, v)))
        edge_groups.setdefault(key_pair, []).append((u, v, key))

    edge_curvatures = {}
    for key_pair, edges in edge_groups.items():
        if len(edges) == 2:
            # If the two edges are opposites (u->v and v->u), assign opposite curvatures.
            (u1, v1, key1), (u2, v2, key2) = edges
            if u1 == v2 and v1 == u2:
                edge_curvatures[(u1, v1, key1)] = 0.2
                edge_curvatures[(u2, v2, key2)] = -0.2
            else:
                for edge in edges:
                    edge_curvatures[edge] = 0.0
        else:
            # For unidirectional edges or other cases, use a straight line.
            for edge in edges:
                edge_curvatures[edge] = 0.0

    # -----------------------
    # Draw the graph manually
    # -----------------------
    fig = plt.figure(figsize=(18, 18))

    # Draw each edge with its assigned curvature.
    for u, v, key in G_multi.edges(keys=True):
        rad = edge_curvatures.get((u, v, key), 0.0)
        prob = G_multi.edges[u, v, key]["prob"]
        if rad > 0:
            edge_color = "magenta"
        elif rad == 0:
            edge_color = "black"
        else:
            edge_color = "green"
        nx.draw_networkx_edges(
            G_multi,
            pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3, rad={abs(rad)}",
            arrowstyle="-|>",
            arrowsize=15,
            edge_color=edge_color,
            alpha=prob,
        )

    # Draw nodes and labels.
    nx.draw_networkx_nodes(G_multi, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G_multi, pos)

    # -----------------------
    # Manually add edge labels
    # -----------------------
    # For each edge, compute a label position that follows the curvature.
    for u, v, key in G_multi.edges(keys=True):
        prob = G_multi.edges[u, v, key]["prob"]
        # Only label edges with non-zero probability.
        if prob == 0:
            label = "0"
        else:
            label = f"{prob:.2f}"
        # Get node positions.
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # Compute the midpoint.
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        # Determine perpendicular offset based on the line from u to v.
        dx = x2 - x1
        dy = y2 - y1
        distance = (dx**2 + dy**2) ** 0.5
        rad = edge_curvatures.get((u, v, key), 0.0)
        if distance == 0:
            offset = (0, 0)
        else:
            # Compute a unit vector perpendicular to the edge.
            offset_x = -dy / distance
            offset_y = -dx / distance
            # Use the curvature (rad) to offset the label.
            # Adjust the scaling factor as needed.
            scale = (-1) ** (label == "0") * distance
            offset = (
                rad * offset_x * scale,
                abs(rad) * offset_y * scale,
            )
        label_pos = (xm + offset[0], ym + offset[1])
        if rad > 0:
            color = "magenta"
        elif rad == 0:
            color = "black"
        else:
            color = "green"
        plt.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=10,
            color=color,
            horizontalalignment="center",
            verticalalignment="center",
            alpha=prob,
        )

    plt.axis("off")
    plt.show()


def hetero_data_equal(data: HeteroData, expected: HeteroData):
    assert isinstance(data, HeteroData) and isinstance(expected, HeteroData)
    assert set(data.node_types) == set(expected.node_types)
    assert set(data.edge_types) == set(expected.edge_types)
    for key in data.node_types:
        dx = data[key].x
        ex = expected[key].x
        if dx.numel() == ex.numel() == 0:
            continue
        if not broadcastable(dx.shape, ex.shape):
            return False
        if not torch.allclose(dx, ex):
            return False
    for edge_type in data.edge_types:
        if not torch.equal(data[edge_type].edge_index, expected[edge_type].edge_index):
            return False
    return True
