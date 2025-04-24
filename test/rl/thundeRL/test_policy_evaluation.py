import itertools
import warnings
from test.fixtures import fresh_flashdrive, medium_blocks  # noqa: F401, F403

import networkx as nx
import torch
from matplotlib import pyplot as plt

import xmimir as xmi
from rgnet.rl.envs import ExpandedStateSpaceEnv
from rgnet.rl.optimality_utils import bellman_optimal_values, optimal_policy
from rgnet.rl.policy_evaluation import (
    PolicyEvaluationMessagePassing,
    mdp_graph_as_pyg_data,
)
from rgnet.rl.reward import UnitReward


def _placeholder_probs(space: xmi.XStateSpace):
    # Contains 0...(|Edges| - 1) as probabilities
    it = itertools.count()
    return tuple(
        torch.Tensor(
            [next(it) for _ in range(space.forward_transition_count(state))],
        )
        for state in space
    )


def test_build_mdp_graph(medium_blocks):
    space: xmi.XStateSpace = medium_blocks[0]
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=0.9),
    )
    env.reset()
    nx_graph = env.to_mdp_graph()
    assert all(
        all(
            out_edge[0] == s.index and out_edge[1] == out_transition.target.index
            for out_edge, out_transition in zip(
                nx_graph.out_edges(nbunch=[i]), space.forward_transitions(s)
            )
        )
        for i, s in enumerate(space)
    )


def test_mdp_graph_as_pyg_data(medium_blocks):
    space: xmi.XStateSpace = medium_blocks[0]
    probs_list = _placeholder_probs(space)
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=0.9),
    )
    env.reset()
    pyg_graph = mdp_graph_as_pyg_data(env.to_mdp_graph(probs_list))
    # Check that the probabilities are stored in the edge_attr
    # Note that we cannot use positional comparison of probabilities stored, as the edges order is not guaranteed, i.e.
    # this is not a valid test:
    assert (pyg_graph.edge_attr[:, 0] == torch.cat(probs_list)).all()
    # Instead, we check that the probabilities are stored in the edge_attr cumulatively and each value is found
    # somewhere (hedge against different terms summing up to the correct value).
    assert pyg_graph.edge_attr[:, 0].sum() == torch.cat(probs_list).sum() and all(
        prob in pyg_graph.edge_attr[:, 0] for prob in torch.cat(probs_list)
    )


def test_mp_on_optimal_medium(medium_blocks):
    space, _, _ = medium_blocks
    gamma = 0.9
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=gamma),
    )
    env.reset()

    value_iteration_mp = PolicyEvaluationMessagePassing(
        gamma, num_iterations=100, difference_threshold=0.001
    )

    optimal_policy_dict = optimal_policy(space)

    def optimal_probs(i, state):
        optimal_action_idx = next(iter(optimal_policy_dict[i]))
        probs = torch.zeros(
            (len(list(space.forward_transitions(state))),),
            dtype=torch.float,
        )
        probs[optimal_action_idx] = 1.0
        return probs

    optimal_policy_probabilities: tuple[torch.Tensor, ...] = tuple(
        optimal_probs(i, s) for (i, s) in enumerate(space)
    )
    graph = env.to_mdp_graph(optimal_policy_probabilities)
    graph = mdp_graph_as_pyg_data(graph)

    values = value_iteration_mp(graph)

    optimal_values = bellman_optimal_values(space, gamma=gamma)

    assert torch.allclose(values, optimal_values, 0.01)


def test_mp_on_faulty_medium(medium_blocks):
    """
    Test that running policy evaluation on a policy that never reaches the goal will yield
    discounted infinite trajectory values for all non-goal states.
    """
    space, _, _ = medium_blocks

    gamma = 0.9
    env = ExpandedStateSpaceEnv(
        space,
        batch_size=torch.Size((1,)),
        reward_function=UnitReward(gamma=gamma),
    )
    env.reset()
    policy_eval_mp = PolicyEvaluationMessagePassing(
        gamma, num_iterations=100, difference_threshold=0.001
    )
    goal_state = next(space.goal_states_iter())
    one_before_goal = list(space.backward_transitions(goal_state))
    assert len(one_before_goal) == 1, (
        "Test assumption violated."
        " Medium blocks problem should only have one goal state with one predecessor."
    )
    one_before_goal = one_before_goal[0].source
    one_before_goal_idx = one_before_goal.index

    # make sure the goal is never reached
    def faulty_probs(i, s):
        nr_transitions = space.forward_transition_count(s)
        probs = torch.rand(
            (nr_transitions,),
            dtype=torch.float,
            generator=torch.Generator().manual_seed(123456789),
        )
        if i == one_before_goal_idx:
            probs = torch.zeros((nr_transitions,), dtype=torch.float)
            for i, succ in enumerate(space.forward_transitions(s)):
                if succ.target != goal_state:
                    probs[i] = 1.0 / nr_transitions
        return probs.abs() / probs.abs().sum()

    probs_list = tuple(faulty_probs(i, s) for (i, s) in enumerate(space))

    graph = env.to_mdp_graph(probs_list)
    graph_data = mdp_graph_as_pyg_data(graph)

    _debug_policy_per_state = {
        node: sum(
            graph_data.edge_attr[graph_data.edge_index[1, :] == node.index, 0].tolist()
        )
        for node in graph.nodes
    }
    if not all(abs(s - 1.0) < 1e-5 for s in _debug_policy_per_state.values()):
        warnings.warn("Policy does not sum up to 1 for all states.")

    # _debug_plotit(graph_data, goal_state)

    values = policy_eval_mp(graph_data)

    # The goal is never reached; therefore, the values for all states should go towards
    # the discounted infinite trajectory length, which is -1 / (1-gamma).
    expected_values = torch.full((len(space),), -1 / (1 - gamma))
    expected_values[goal_state.index] = 0.0
    assert torch.allclose(values, expected_values, atol=0.01)
    # We can never go beyond -1 / (1 gamma).
    #
    assert (values >= -10).all()


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
