import itertools
from typing import Callable, Sequence

import networkx as nx
import torch
import torch_geometric.data
from torch import Tensor
from torch.nn.functional import l1_loss
from torch_geometric.nn.conv import MessagePassing

from rgnet.rl.envs import ExpandedStateSpaceEnv
from xmimir import XState, XTransition


def build_mdp_graph(
    env: ExpandedStateSpaceEnv,
    transition_probabilities: (
        Sequence[torch.FloatTensor | torch.Tensor]
        | Callable[[XState, Sequence[XTransition]], Sequence[float]]
        | None
    ) = None,
    use_space_directly: bool = True,
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
      - the action (edge attribute "action")

    Parameters
    ----------
    env: ExpandedStateSpaceEnv
        The environment to build the graph from.
    transition_probabilities: (
           Sequence[torch.FloatTensor | torch.Tensor]
           | Callable[[XState, Sequence[XTransition]], Sequence[float]]
           | None
    )
           The transition probabilities for each state. If None, uniform transition probabilities are used.
    use_space_directly: bool
           If True, the state space is used directly.
           Otherwise, the environment is used to traverse the space and generate the graph.
           This is significantly faster for large state spaces, since default .traverse() functions
           merely replicate the state space as a tensordict.
    """
    mdp_graph = nx.MultiDiGraph()

    # we sidestep the env-logic of using the .traverse() function to have significantly faster graph generation
    if use_space_directly:
        env.reset()
        space = env.active_instances[0]
        states = space
        instances = [space] * len(space)
        transitions = (env.get_applicable_transitions((state,))[0] for state in space)
    else:
        traversal_td = env.traverse()[0]
        states = traversal_td["state"]
        instances = traversal_td["instance"]
        transitions = traversal_td["transitions"]

    if transition_probabilities is None:

        def transition_probs(s: XState, ts: Sequence[XTransition]):
            n_trans = len(ts)
            if n_trans == 0:
                raise ValueError(
                    "Given state has no transitions. "
                    "At least a self-loop transition (for e.g. dead-ends/goals) is expected."
                )
            return torch.ones((n_trans,), dtype=torch.float) / n_trans

    elif isinstance(transition_probabilities, Sequence):
        assert len(transition_probabilities) == len(states)

        def transition_probs(s: XState, ts: Sequence[XTransition]):
            return transition_probabilities[s.index]

    else:
        transition_probs = transition_probabilities
    for state, instance in zip(states, instances):
        node_type = (
            "goal"
            if env.is_goal(instance, state)
            else ("initial" if instance.initial_state == state else "default")
        )
        mdp_graph.add_node(
            state,
            ntype=node_type,
            dist=instance.goal_distance(state),
        )
    running_transition_idx = itertools.count(0)
    for state, instance, state_transitions in zip(states, instances, transitions):
        t_probs: Sequence[float] = transition_probs(state, state_transitions)
        reward, _ = env.get_reward_and_done(
            state_transitions, instances=[instance] * len(state_transitions)
        )
        for t_idx, t in enumerate(state_transitions):
            mdp_graph.add_edge(
                t.source,
                t.target,
                action=t.action,
                reward=reward[t_idx],
                probs=t_probs[t_idx],
                idx=next(running_transition_idx),
            )
    mdp_graph.graph["gamma"] = env.reward_function.gamma
    return mdp_graph


def mdp_graph_as_pyg_data(nx_state_space_graph: nx.DiGraph):
    """
    Convert the networkx graph into a directed pytorch_geometric graph.
    The transition probabilities are stored in edge_attr[:, 0].
    The reward for each transition is stored in edge_attr[:, 1].
    The node features are stored as usual in graph.x.
    The first dimension is the node value (starting with 0).
    The second node feature dimension is one, if the node is a goal state.
    """
    pyg_graph = torch_geometric.utils.from_networkx(
        nx_state_space_graph, group_edge_attrs=["probs", "reward", "idx"]
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
        if attr["ntype"] == "goal":
            is_goal_state[i] = True
            _, _, goal_reward[i] = next(
                iter(nx_state_space_graph.out_edges(node, data="reward"))
            )

    pyg_graph.goals = torch.tensor(
        is_goal_state,
        dtype=torch.bool,
    )
    # goal states have the value of their reward (typically 0, but could be arbitrary),
    # rest is initialized to 0.
    pyg_graph.x = torch.where(
        pyg_graph.goals,
        torch.tensor(goal_reward, dtype=torch.float),
        torch.zeros((pyg_graph.num_nodes,)),
    )
    if hasattr(nx_state_space_graph.graph, "gamma"):
        pyg_graph.gamma = nx_state_space_graph.graph["gamma"]
    return pyg_graph


# MessagePassing interface defines message_and_aggregate and edge_update, which are
# marked abstract but should only be overwritten if needed.
# noinspection PyMethodOverriding, PyAbstractClass
class PolicyEvaluationMessagePassing(MessagePassing):
    """
    Implements Policy evaluation as a Pytorch Geometric message passing function.
    Can be executed on cpu or an accelerator.

    Note: Policy evaluation follows the Bellman equation:

    .. math::
        V(s) = \sum_{a} \sum_{s'} \pi(a|s) P(s'|s,a) [R(s,a,s') + \gamma V(s')]
             = \sum_{s'} \pi(s'|s) [R(s,s') + \gamma V(s')]

    where :math:`V(s)` is the value of state :math:`s`, :math:`P(s'|s,a)` is the transition probability from state
    :math:`s` to state :math:`s'` from action :math:`a`, and :math:`R(s,a,s')` is the reward for taking action :math:`a`
    in state :math:`s` and ending up in state :math:`s'`.
    Our MDPs are assumed deterministic, thus actions coincide with successor states (second line).
    In this setting we consider policies to be mappings of states to successor states (and not actions),
    effectively collapsing the transition probabilities and the policy into one.

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
        *,
        aggr: str = "sum",
    ):
        super().__init__(aggr=aggr, node_dim=-1, flow="target_to_source")
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
            # We have to ensure that goal states stay at the given goal value (0.0 often, but decided by reward func).
            # environment transitions from goal states would terminate.
            values = torch.where(goal_states, values, new_values)
        data.x = values
        return values

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        transition_prob, reward = edge_attr[:, 0], edge_attr[:, 1]
        return transition_prob * (reward + self.gamma * x_j)


# noinspection PyMethodOverriding, PyAbstractClass
class ValueIterationMessagePassing(PolicyEvaluationMessagePassing):
    """
    Implements Value Iteration as a Pytorch Geometric message passing function.
    Can be executed on cpu or an accelerator.

    Note: Value Iteration is follows the formula:

    .. math::
        V(s) = \max_{a} \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
             = \max_{s'} [R(s,s') + \gamma V(s')]

    where V(s) is the value of state s, P(s'|s,a) is the transition probability from state s to state s'.
    Therefore, we can reuse the same message passing module as for policy evaluation.
    The only difference is that we need to use the max operator instead of the sum operator and implicitly assume that
    the policy in use is:

    .. math::
        \pi(s'|s) = \argmax_{s'} [R(s,s') + \gamma V(s')]

    Hence, to emulate this we need the transition "probabilities" to be merely a weight of 1.0 for all edges.
    We enforce this by setting the transition probabilities to 1.0 actively in the forward method.

    Assumes the input Data object has the following attributes:
    - edge_attr: Tensor of shape [num_edges, 2] containing transition probabilities and rewards
                 edge_attr[:, 0] = transition probabilities
                 edge_attr[:, 1] = rewards
    - node_attr: Tensor of shape [num_nodes] containing the initial state values
    - goals: BoolTensor of shape [num_nodes] with goals[i] == space.is_goal_state(space.get_states()[i])
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs["aggr"] = "max"
        super().__init__(*args, **kwargs)

    def forward(self, data: torch_geometric.data.Data) -> Tensor:
        data.edge_attr[:, 0] = 1.0
        return super().__call__(data)
