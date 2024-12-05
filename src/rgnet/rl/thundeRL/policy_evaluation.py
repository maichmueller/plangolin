from typing import List

import networkx as nx
import pymimir as mi
import torch
import torch_geometric.data
from torch import Tensor
from torch.nn.functional import l1_loss
from torch_geometric.nn.conv import MessagePassing


def build_mdp_graph_with_prob(
    state_space: mi.StateSpace,
    transition_probabilities: List[torch.FloatTensor | torch.Tensor],
) -> nx.DiGraph:
    """
    Encode the state space as networkx graph. Each state is a node and each transition
    corresponds to an edge. The states are encoded as unique integers.
    The rewards are analog as in PlanningEnvironment such that goal states have a value of 0.
    The graph also contains information about:
     - whether a state is initial or goal state (node attribute "ntype"),
     - the distance to the goal state (node attribute "dist"),
     - the reward for each edge/transition which is -1 for each transition from each non-goal state
        (edge attribute "reward"),
     - the transition probabilities (edge attribute "probs"),
     - the action schema name (edge attribute "action")
    """
    mdp_graph = nx.DiGraph()
    for idx, state in enumerate(state_space.get_states()):
        node_type = (
            "goal"
            if state_space.is_goal_state(state)
            else ("initial" if state_space.get_initial_state() == state else "default")
        )
        assert state_space.get_unique_id(state) == idx
        mdp_graph.add_node(
            state_space.get_unique_id(state),
            ntype=node_type,
            dist=state_space.get_distance_to_goal_state(state),
        )
    for state_idx, state in enumerate(state_space.get_states()):
        t_probs: List[float] = transition_probabilities[state_idx].tolist()
        reward = 0.0 if state_space.is_goal_state(state) else -1.0
        for t_idx, t in enumerate(state_space.get_forward_transitions(state)):
            mdp_graph.add_edge(
                state_space.get_unique_id(t.source),
                state_space.get_unique_id(t.target),
                action=t.action.schema.name,
                reward=reward,
                probs=t_probs[t_idx],
            )
    return mdp_graph


def mdp_graph_as_pyg_data(nx_state_space_graph: nx.DiGraph):
    """
    Convert the networkx graph into a directed pytorch graph.
    The transition probabilities are stored in edge_attr[:, 0].
    The reward for each transition is stored in edge_attr[:, 1].
    The node features are stored as usual in graph.x.
    The first dimension is the node value (starting with 0).
    The second node feature dimension is one, if the node is a goal state.
    """
    pyg_graph = torch_geometric.utils.from_networkx(
        nx_state_space_graph, group_edge_attrs=["probs", "reward"]
    )
    pyg_graph.x = torch.zeros((pyg_graph.num_nodes,))  # start with values of zero
    is_goal_state = torch.tensor(
        [
            True if attr["ntype"] == "goal" else False
            for node, attr in nx_state_space_graph.nodes.data()
        ],
        dtype=torch.bool,
    )
    pyg_graph.goals = is_goal_state
    return pyg_graph


# MessagePassing interface defines message_and_aggregate and edge_update, which are
# marked abstract but should only be overwritten if needed.
# noinspection PyMethodOverriding, PyAbstractClass
class PolicyEvaluationMessagePassing(MessagePassing):
    """
    Implements Policy evaluation as a Pytorch Geometric message passing function.
    Can be executed on cpu or an accelerator.

    Assumes the input Data object has the following attributes:
    - edge_attr: Tensor of shape [num_edges, 2] containing transition probabilities and rewards
                 edge_attr[:, 0] = transition probabilities
                 edge_attr[:, 1] = rewards
    - node_attr: Tensor of shape [num_nodes] containing the initial state values
    - goals: BoolTensor of shape [num_nodes] with goals[i] == space.is_goal_state(space.get_states()[i])
    """

    def __init__(
        self,
        gamma: float,
        num_iterations: int = 1_000,
        difference_threshold: float = 0.001,
    ):
        super().__init__(aggr="sum", node_dim=-1, flow="target_to_source")
        self.register_buffer("gamma", torch.as_tensor(gamma))
        self.num_iterations = num_iterations
        self.difference_threshold = difference_threshold

    def forward(self, data: torch_geometric.data.Data) -> Tensor:
        assert isinstance(data.x, torch.Tensor)
        assert data.x.shape == data.goals.shape
        assert data.edge_attr.ndim == 2
        assert data.edge_index.shape[-1] == data.edge_attr.shape[0]
        values = data.x
        goal_states: torch.Tensor | torch.BoolTensor = data.goals
        for _ in range(self.num_iterations):
            new_values: torch.Tensor = self.propagate(
                edge_index=data.edge_index, edge_attr=data.edge_attr, x=values
            )
            if l1_loss(values, new_values) < self.difference_threshold:
                break
            # We have to ensure that goal states stay at 0.0
            # In en environment transitions from goal states would terminate.
            new_values.masked_fill_(goal_states, 0.0)
            values = new_values
        return values

    def message(self, x_j: Tensor, edge_attr) -> Tensor:
        transition_prob, reward = edge_attr[:, 0], edge_attr[:, 1]
        return transition_prob * (reward + self.gamma * x_j)
