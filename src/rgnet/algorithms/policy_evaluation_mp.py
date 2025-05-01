from typing import Optional

import torch
import torch_geometric as pyg
from torch import Tensor
from torch.nn.functional import l1_loss
from torch_geometric.nn.conv import MessagePassing

from rgnet.utils.reshape import unsqueeze_right_like


# MessagePassing interface defines message_and_aggregate and edge_update, which are
# marked abstract but should only be overwritten if needed.
# noinspection PyMethodOverriding, PyAbstractClass
class PolicyEvaluationMP(MessagePassing):
    r"""
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
                 edge_attr[:, 0] = rewards
                 edge_attr[:, 1] = transition probabilities
    - node_attr: Tensor of shape [num_nodes] containing the initial state values
    - goals: BoolTensor of shape [num_nodes] with goals[i] == space.is_goal_state(space.get_states()[i])
    """

    default_attr_name = "policy_values"

    def __init__(
        self,
        gamma: float,
        num_iterations: int = 1_000,
        difference_threshold: float | None = 0.001,
        *,
        attr_name: str | None = None,
        aggr: str = "sum",
    ):
        super().__init__(aggr=aggr, node_dim=-1, flow="target_to_source")
        self.register_buffer("gamma", torch.as_tensor(gamma))
        self.num_iterations = num_iterations
        self.difference_threshold = difference_threshold
        self.attr_name = attr_name or self.default_attr_name

    def forward(self, data: pyg.data.Data) -> Tensor:
        if hasattr(data, self.attr_name):
            return getattr(data, self.attr_name)
        values = self._forward(data)
        setattr(data, self.attr_name, values)
        return values

    def _forward(self, data):
        assert isinstance(data.x, torch.Tensor)
        assert data.edge_attr.ndim == 2
        assert data.edge_index.shape[-1] == data.edge_attr.shape[0]
        values = self._init_values(data)
        for _ in range(self.num_iterations):
            new_values: torch.Tensor = self.propagate(
                edge_index=data.edge_index, edge_attr=data.edge_attr, x=values
            )
            if (
                self.difference_threshold is not None
                and l1_loss(values, new_values) < self.difference_threshold
            ):
                break
            values = self._apply_new_values(data, values, new_values)
        return values

    def _init_values(self, data):
        return torch.zeros_like(data.x)

    def _apply_new_values(self, data, prev_values, new_values):
        """
        We have to ensure that goal states stay at the given goal value (0.0 often, but decided by reward func).
        Environment transitions from goal states would terminate.
        """
        assert data.goals.shape == prev_values.shape
        return torch.where(data.goals, prev_values, new_values)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        reward, transition_prob = edge_attr[:, 0], edge_attr[:, 1]
        return transition_prob * (reward + self.gamma * x_j)


# noinspection PyMethodOverriding, PyAbstractClass
class ValueIterationMP(PolicyEvaluationMP):
    r"""
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
    We enforce this by setting the transition probabilities to 1.0 actively in the message method.

    Assumes the input Data object has the following attributes:
    - edge_attr: Tensor of shape [num_edges, 2] containing rewards
                 edge_attr[:, 0] = rewards
    - node_attr: Tensor of shape [num_nodes] containing the initial state values
    - goals: BoolTensor of shape [num_nodes] with goals[i] == space.is_goal_state(space.get_states()[i])
    """

    default_attr_name = "bellman_optimal_values"

    def __init__(
        self,
        *args,
        aggr: str = "max",
        **kwargs,
    ):
        super().__init__(*args, aggr=aggr, **kwargs)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        reward = unsqueeze_right_like(edge_attr[:, 0], x_j)
        return reward + self.gamma * x_j


# noinspection PyMethodOverriding, PyAbstractClass
class OptimalPolicyMP(ValueIterationMP):
    r"""
    Implements Optimal Policy as a Pytorch Geometric message passing function.
    Can be executed on cpu or an accelerator.

    Note: Optimal Policy follows the formula:

    .. math::
        \pi(s'|s) = \argmax_{s'} [R(s,s') + \gamma V(s')]

    where V(s) is the value of state s. In a deterministic MDP, actions coincide with successor states.
    """

    default_attr_name = "optimal_policy"

    def __init__(
        self,
        *args,
        aggr=None,
        value_iteration_mp: Optional[ValueIterationMP] = None,
        num_iterations: int = 1,  # a single iteration is enough
        difference_threshold: float | None = None,  # not used
        **kwargs,
    ):
        super().__init__(
            *args,
            aggr=aggr,
            num_iterations=num_iterations,
            difference_threshold=difference_threshold,
            **kwargs,
        )
        self.value_iteration_mp = value_iteration_mp or ValueIterationMP(self.gamma)

    def _forward(self, data: torch_geometric.data.Data) -> Tensor:
        """
        Computes the optimal policy for the given data object.

        Assumes the input Data object has the following attributes:
        - edge_attr: Tensor of shape [num_edges, 2] containing rewards in the first entry
                     edge_attr[:, 0] = rewards
        """

    def _init_values(self, data):
        return self.value_iteration_mp(data)

    def _apply_new_values(self, data, prev_values, new_values):
        return new_values

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return torch.empty_like(inputs).scatter_reduce(
            0, index, inputs, reduce="amax", include_self=False
        )
