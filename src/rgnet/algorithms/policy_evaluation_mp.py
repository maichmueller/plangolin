import itertools
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
        flow="target_to_source",
        attr_name: str | None = None,
        aggr: str = "sum",
    ):
        super().__init__(aggr=aggr, node_dim=0, flow=flow)
        self.register_buffer("gamma", torch.as_tensor(gamma))
        self.num_iterations = num_iterations
        self.difference_threshold = difference_threshold
        self.attr_name = attr_name or self.default_attr_name

    def forward(self, data: pyg.data.Data) -> Tensor:
        if hasattr(data, self.attr_name):
            return getattr(data, self.attr_name)
        values = self._iterate(data)
        setattr(data, self.attr_name, values)
        self._reset()
        return values

    def _reset(self):
        """
        Cleanup method to be called after the forward pass.
        """
        pass

    def _iterate(self, data):
        assert isinstance(data.x, torch.Tensor)
        assert data.edge_attr.ndim >= 2
        assert data.edge_index.shape[-1] == data.edge_attr.shape[0]
        features = self._init_features(data)
        edge_indices = data.edge_index
        layer_count = itertools.count()
        while True:
            new_features: torch.Tensor = self.propagate(
                edge_index=edge_indices, edge_attr=data.edge_attr, x=features, data=data
            )
            if self._break_condition(
                next(layer_count), features, new_features, data=data
            ):
                break
            features = new_features
        return features

    def _break_condition(
        self, layer_nr: int, features: Tensor, new_features: Tensor, **kwargs
    ):
        if layer_nr >= self.num_iterations:
            return True
        if self.difference_threshold is None:
            return False
        feature_infs = torch.isinf(features)
        new_feature_infs = torch.isinf(new_features)
        both_not_infs = ~(feature_infs & new_feature_infs)
        bc_features, bc_new_features = torch.broadcast_tensors(features, new_features)
        return (
            l1_loss(bc_features[both_not_infs], bc_new_features[both_not_infs])
            < self.difference_threshold
        )

    def _init_features(self, data):
        return torch.zeros_like(data.x)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        reward, transition_prob = edge_attr[:, 0], edge_attr[:, 1]
        reward, transition_prob = (
            unsqueeze_right_like(reward, x_j),
            unsqueeze_right_like(transition_prob, x_j),
        )
        return transition_prob * (reward + self.gamma * x_j)

    def update(self, inputs: Tensor, x: Tensor, data: pyg.data.Data) -> Tensor:
        return torch.where(unsqueeze_right_like(data.goals, inputs), x, inputs)


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
        aggr: str | None = "max",
        **kwargs,
    ):
        super().__init__(*args, aggr=aggr, **kwargs)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        reward = unsqueeze_right_like(edge_attr[:, 0], x_j)
        return reward + self.gamma * x_j

    def update(self, inputs: Tensor) -> Tensor:
        return inputs


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

    def _iterate(self, data: pyg.data.Data) -> Tensor:
        """
        Computes the optimal policy for the given data object.

        Assumes the input Data object has the following attributes:
        - edge_attr: Tensor of shape [num_edges, 2] containing rewards in the first entry
                     edge_attr[:, 0] = rewards
        """

    def _init_features(self, data):
        return self.value_iteration_mp(data)

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


# noinspection PyMethodOverriding, PyAbstractClass
class OptimalAtomValuesMP(ValueIterationMP):
    r"""
    Implements Atom Distance as a Pytorch Geometric message passing function.
    Can be executed on cpu or an accelerator.

    Note: Atom Distance follows the formula:

    .. math::
        \forall p not true in s: d(s, p) = \min_{s'} \{ d(s, s') : p is true in s' \}

    where d(s, p) is the distance from state s to atom p. In a deterministic MDP, actions coincide with successor states.

    We expect an incoming data object to either have pre-initialized x values or to have the attribute atoms_per_state,
    which is a list[list[XAtom]] of atoms that are true in the respective state.
    The length of this attribute would have to be the number of states to propagate messages for.
    """

    default_attr_name = "atom_values"

    def __init__(
        self,
        gamma: float = 1.0,
        num_iterations: int = 1_000,
        difference_threshold: float | None = 1e-5,
        *,
        known_atom_reward: float = 0.0,
        atom_to_index_map: dict[str, int],
        aggr: str = "max",
        flow="target_to_source",
        **kwargs,
    ):
        super().__init__(
            gamma,
            num_iterations,
            difference_threshold,
            aggr=aggr,
            flow=flow,
            **kwargs,
        )
        match aggr:
            case "min":
                self.updater_func = torch.min
                self.sign = 1.0
            case "max":
                self.updater_func = torch.max
                self.sign = -1.0
            case _:
                raise ValueError(
                    f"Unsupported aggregation method '{aggr}'. Supported methods are 'min' (costs) and 'max' (rewards)."
                )
        self.init_reward = self.sign * torch.inf
        self.atom_to_index = atom_to_index_map
        self.num_atoms = max(atom_to_index_map.values()) + 1
        self.known_atom_reward = known_atom_reward

    def _init_features(self, data):
        if not hasattr(data, "atoms_per_state"):
            raise ValueError(
                "Data object must have 'atoms_per_state' attribute if features is not pre-initialized."
            )
        num_states = data.num_nodes
        num_atoms = self.num_atoms
        device = data.edge_index.device
        features = torch.full(
            (num_states, num_atoms), self.init_reward, device=device, dtype=torch.float
        )
        for state_idx, atom_iterable in enumerate(data.atoms_per_state):
            atom_list = list(atom_iterable)
            if not atom_list:
                continue
            features[
                state_idx, [self.atom_to_index[str(atom)] for atom in atom_list]
            ] = 0.0
        return features

    def message(
        self,
        x_j: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        reward = unsqueeze_right_like(edge_attr[..., 0], x_j)
        return reward + self.gamma * x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """
        PyG's aggregation module calls scatter into a new torch.zeros tensor. Any target index not
        mentioned in `index` will not aggregate any values and will hence be set to the existing
        value at the tensor to write to, i.e., the 0-tensor. This will see any values that reflect
        our init_reward (i.e., +- inf) be set to 0.0, which casts errors into future message-passes.
        By ensuring that every index is mentioned in the aggregation with a value from our `init_reward`,
        we can ensure that the aggregation will not be set to 0.0 by default.
        Note this only works for the min/max aggregation, as the sum aggregation would result in inf's then.
        """
        extended_inputs = torch.cat(
            (
                inputs,
                torch.full(
                    (dim_size, inputs.size(1)),
                    self.init_reward,
                    device=inputs.device,
                    dtype=inputs.dtype,
                ),
            ),
            dim=0,
        )
        extended_index = torch.cat(
            (index, torch.arange(dim_size, device=index.device, dtype=torch.int)), dim=0
        )
        return super().aggregate(extended_inputs, extended_index, ptr, dim_size)

    def update(self, inputs: Tensor, x: Tensor = None) -> Tensor:
        return self.updater_func(inputs, x)


if __name__ == "__main__":
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node(0, ntype="state")
    graph.add_node(1, ntype="state")
    graph.add_node(2, ntype="state")
    graph.add_edge(0, 1, reward=1.0, probs=0.5, idx=0)
    graph.add_edge(1, 2, reward=1.0, probs=0.5, idx=1)
    graph.add_edge(2, 0, reward=1.0, probs=0.5, idx=2)

    atom_to_index_map = {
        "t": 0,
        "q": 1,
        "p": 2,
    }
    from rgnet.algorithms import mdp_graph_as_pyg_data

    pyg_graph = mdp_graph_as_pyg_data(graph)
    pyg_graph.atoms_per_state = [["p"], ["q"], ["t"]]
    mp_module = OptimalAtomValuesMP(atom_to_index_map=atom_to_index_map, aggr="min")
    mp_module(pyg_graph)
    final_distances = pyg_graph[OptimalAtomValuesMP.default_attr_name]
    expected = torch.tensor([[2, 1, 0], [1, 0, 2], [0, 2, 1]], dtype=torch.float)
    assert final_distances.shape == expected.shape
    assert torch.allclose(final_distances, expected)
