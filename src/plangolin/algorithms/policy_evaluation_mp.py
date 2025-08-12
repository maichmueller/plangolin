import itertools
from typing import Optional

import torch
import torch_geometric as pyg
from torch import Tensor
from torch.nn.functional import l1_loss
from torch_geometric.nn.conv import MessagePassing

from plangolin.models.mixins import DeviceAwareMixin
from plangolin.utils.reshape import unsqueeze_right_like


# MessagePassing interface defines message_and_aggregate and edge_update, which are
# marked abstract but should only be overwritten if needed.
# noinspection PyMethodOverriding, PyAbstractClass
class PolicyEvaluationMP(DeviceAwareMixin, MessagePassing):
    r"""
    Policy evaluation as a PyTorch Geometric `MessagePassing` layer.

    Implements the Bellman expectation backup

    .. math::
        V(s) = \sum_{s'} \pi(s'\mid s)\,[R(s,s') + \gamma V(s')]

    assuming deterministic MDPs where actions coincide with successor states (policy maps states to successors).

    **Input requirements**
    - `data.edge_attr`: tensor with at least two columns. Only `edge_attr[:, 0]` (reward) and `edge_attr[:, 1]`
      (transition probability) are used; additional columns are ignored.
    - `data.x`: tensor of shape `[num_nodes, ...]` containing the current value estimate (any shape that broadcasts
      across messages is supported). By default, this module initializes `x` to zeros.
    - `data.goals`: `BoolTensor[num_nodes]` indicating terminal/goal states. Goal states keep their current value.

    The resulting values are returned and, if `cache=True`, stored on the `Data` object under
    `self.attr_name` (default: `"policy_values"`).
    """

    default_attr_name = "policy_values"

    def __init__(
        self,
        gamma: float,
        num_iterations: int = 1_000,
        difference_threshold: float | None = 0.001,
        *,
        flow="target_to_source",
        attr_name: str = default_attr_name,
        aggr: str = "sum",
    ):
        super().__init__(aggr=aggr, node_dim=0, flow=flow)
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.difference_threshold = difference_threshold
        self.attr_name = attr_name

    def forward(self, data: pyg.data.Data, cache: bool = True) -> Tensor:
        """
        Run fixed‑point iteration until the stopping criterion is met, optionally caching the result on `data`.

        Parameters
        ----------
        data : pyg.data.Data
            Graph with `edge_index`, `edge_attr`, `x`, and `goals`.
        cache : bool, default True
            If `True`, store the result under `data.<attr_name>` and return the cached value on subsequent calls.
        """
        if cache and hasattr(data, self.attr_name):
            return getattr(data, self.attr_name)
        values = self._iterate(data)
        if cache:
            setattr(data, self.attr_name, values)
        self._reset()
        return values

    def _reset(self):
        """
        Cleanup method to be called after the forward pass.
        """
        pass

    def _iterate(self, data):
        """
        Perform iterative message passing until `num_iterations` is reached or the L1 change falls below
        `difference_threshold` (if set).
        """
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
        """
        Stopping rule for the value‑iteration loop.

        Returns `True` if the maximum number of iterations is reached, or if the mean absolute difference between
        consecutive value tensors (ignoring `inf` at matching positions) is below `difference_threshold`.
        """
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
        """
        Initialize the value tensor. Default: zeros like `data.x`.
        """
        return torch.zeros_like(data.x)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Bellman expectation message from successor `j` to source `i`:

        `P(s'\mid s) * (R(s,s') + gamma * V(s'))`

        Expects `edge_attr[:, 0]` to hold rewards and `edge_attr[:, 1]` transition probabilities.
        """
        reward, transition_prob = edge_attr[:, 0], edge_attr[:, 1]
        reward, transition_prob = (
            unsqueeze_right_like(reward, x_j),
            unsqueeze_right_like(transition_prob, x_j),
        )
        return transition_prob * (reward + self.gamma * x_j)

    def update(self, inputs: Tensor, x: Tensor, data: pyg.data.Data) -> Tensor:
        """
        Keep existing values at goal states; otherwise accept the aggregated backup.
        """
        return torch.where(unsqueeze_right_like(data.goals, inputs), x, inputs)


# noinspection PyMethodOverriding, PyAbstractClass
class ValueIterationMP(PolicyEvaluationMP):
    r"""
    Value iteration as a PyTorch Geometric `MessagePassing` layer.

    Implements the Bellman optimality backup

    .. math::
        V(s) = \max_{s'} [R(s,s') + \gamma V(s')]

    by using `aggr='max'` and messages of the form `R(s,s') + \gamma V(s')`. Transition probabilities are
    ignored for this optimal-control case.

    **Input requirements**
    - `data.edge_attr`: only `edge_attr[:, 0]` (reward) is used; other columns are ignored.
    - `data.x`: initial value estimate per node (default initialization is zeros).
    - `data.goals`: goal mask used by `PolicyEvaluationMP.update` (goal nodes keep their value).

    The resulting values are returned and, if `cache=True`, stored under `"bellman_optimal_values"` by default.
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
    Greedy policy extraction scaffolding based on one Bellman‑optimality backup.

    This class initializes values using a provided `ValueIterationMP` (or a default instance), and performs a single
    max‑aggregation backup. **It currently returns the per‑state maximal action value, not the argmax indices**. To obtain
    the actual greedy policy (successor selection), track argmax indices during aggregation or run a separate selection
    step over outgoing edges using the same scores.

    Parameters
    ----------
    value_iteration_mp : ValueIterationMP, optional
        Module used to compute/initialize the value function prior to the greedy backup.
    num_iterations : int, default 1
        A single iteration is sufficient for the greedy backup.
    difference_threshold : float | None, default None
        Unused in this class.
    """

    default_attr_name = "optimal_policy"

    def __init__(
        self,
        value_iteration_mp: Optional[ValueIterationMP] = None,
        *args,
        **kwargs,
    ):
        super().__init__(1.0)
        self.value_iteration_mp = value_iteration_mp or ValueIterationMP(
            *args,
            **kwargs,
        )
        self._is_first_iter = True

    def _reset(self):
        """
        Reset the internal state of the module.
        """
        self._is_first_iter = True

    def _break_condition(
        self, layer_nr: int, features: Tensor, new_features: Tensor, **kwargs
    ) -> bool:
        # we only run one iteration, so we always break
        first_iter = self._is_first_iter
        self._is_first_iter = False
        return not first_iter

    def _init_features(self, data):
        return self.value_iteration_mp(data)

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """
        Return per-node argmax **global message indices** for the backup scores in `inputs`.

        Two passes:
          1) node-wise max value;
          2) among edges equal to that max, pick smallest global edge id (stable argmax).
        Nodes with no incoming messages return -1.
        """
        if dim_size is None:
            raise ValueError(
                "dim_size must be provided to compute per-node argmax indices."
            )

        # this is the PyG way to compute argmax indices using pytorch_scatter's scatter_max. Unfortunately, pytorch
        # scatter does not provide any argreduce functionality, so we have to implement it ourselves. In the future,
        # I hope to replace the implementation below with a line like this from pure pytorch.
        # TODO: replace with scatter_argmax or similar once pytorch supports argmax indices.
        # _, argmax = scatter_max(inputs, index, dim=0, dim_size=dim_size)
        # return argmax

        # Accept [E] or [E,1] inputs -> squeeze to [E]
        if inputs.ndim == 2 and inputs.size(1) == 1:
            vals = inputs.squeeze()
        elif inputs.ndim == 1:
            vals = inputs
        else:
            raise ValueError(
                f"Expected scalar per-edge inputs; got shape {tuple(inputs.shape)}"
            )

        # max per node
        max_per_node = torch.full(
            (dim_size,),
            float("-inf"),
            dtype=vals.dtype,
            device=vals.device,
        )
        max_per_node.scatter_reduce_(0, index, vals, reduce="amax", include_self=True)

        # argmax via amin of masked edge ids
        is_max = vals == max_per_node.index_select(0, index)
        edge_ids = torch.arange(vals.numel(), device=index.device, dtype=torch.long)
        long_max = torch.iinfo(torch.long).max
        masked_ids = torch.where(is_max, edge_ids, torch.full_like(edge_ids, long_max))

        arg_idx = torch.full(
            (dim_size,), long_max, dtype=torch.long, device=index.device
        )
        arg_idx.scatter_reduce_(0, index, masked_ids, reduce="amin", include_self=True)

        # no-incoming edge -> -1
        no_incoming = torch.isinf(max_per_node) & (max_per_node < 0)

        # Convert global edge ids to per-state local ids (0..deg(s)-1)
        out = arg_idx.clone()
        out[no_incoming] = -1
        valid = ~no_incoming
        # Convert global edge ids to per-state local ids (0..deg(s)-1)
        if valid.any():
            if ptr is not None:
                # Fast path with CSR pointers: local = global_edge_id - ptr[state]
                state_ids = torch.arange(dim_size, device=out.device)
                out[valid] = out[valid] - ptr[state_ids[valid]].to(out.dtype)
            else:
                # Derive local positions per edge even if edges are not grouped by state
                nr_transitions = vals.numel()
                order = torch.argsort(index, stable=True)
                sorted_idx = index[order]
                pos_sorted = torch.arange(nr_transitions, device=index.device)
                first_pos_per_node = torch.full(
                    (dim_size,), nr_transitions, dtype=torch.long, device=index.device
                )
                first_pos_per_node.scatter_reduce_(
                    0, sorted_idx, pos_sorted, reduce="amin", include_self=True
                )
                local_sorted = pos_sorted - first_pos_per_node.index_select(
                    0, sorted_idx
                )
                local_per_edge = torch.empty(
                    nr_transitions, dtype=torch.long, device=index.device
                )
                local_per_edge[order] = local_sorted
                out[valid] = local_per_edge[out[valid]]
        return out


# noinspection PyMethodOverriding, PyAbstractClass
class OptimalAtomValuesMP(ValueIterationMP):
    r"""
    Propagate per‑state values for propositional atoms using optimal backups.

    Let `x[s, p]` denote the value (e.g., distance or reward) of making atom `p` true starting from state `s`.
    For each atom `p` that is already true in `s`, `x[s, p] = 0`. For other atoms, messages apply

    .. math::
        x[s, p] = \operatorname{agg}_{s' \in N(s)} \bigl( R(s,s') + \gamma\, x[s', p] \bigr),

    where `agg` is `min` (costs) or `max` (rewards). Unreachable atoms remain at `+/- inf` depending on the choice of
    aggregation.

    **Input requirements**
    - If `data.x` is not pre‑initialized to shape `[num_states, num_atoms]`, provide `data.atoms_per_state`, a sequence of
      iterables listing atoms true in each state. The mapping from atom to column index is given by `atom_to_index_map`.

    Parameters
    ----------
    gamma : float, default 1.0
        Discount factor.
    num_iterations : int, default 1000
        Maximum number of iterations.
    difference_threshold : float | None, default 1e-5
        L1 threshold for early stopping; set `None` to disable.
    known_atom_reward : float, default 0.0
        Reserved for future use (e.g., shaping known atoms); currently unused.
    atom_to_index_map : dict[str, int]
        Maps `str(atom)` to column index in the value matrix.
    aggr : {"min", "max"}, default "max"
        Aggregation direction: `"min"` for costs, `"max"` for rewards.
    flow : {"target_to_source", "source_to_target"}
        Message flow passed to `MessagePassing`.
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
        # Note: theoretically, we could set this to 1 / 1 - gamma for the infinite plan length dead-end case.
        # In this case we would implicitly assume that the rewards stored in the incoming data are uniform step rewards.
        # This limitation is unnecessarily restrictive, as we can simply use the init_reward value as value to pad.
        # A token value of +- inf be easiest for later masking of unreachable atoms.
        self.init_reward = self.sign * torch.inf
        self.atom_to_index = atom_to_index_map
        self.num_atoms = max(atom_to_index_map.values()) + 1
        self.known_atom_reward = known_atom_reward

    def _init_features(self, data):
        if not hasattr(data, "atoms_per_state"):
            raise ValueError(
                "Data object must have 'atoms_per_state' attribute if `features` is not pre-initialized."
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
        Ensure that nodes with no incoming messages keep their sentinel value (±inf).

        PyG aggregates into a fresh zeros tensor; indices that receive no messages would default to `0.0`. We append one
        synthetic message per node equal to the sentinel `init_reward`, so min/max aggregation preserves the intended value
        for isolated nodes. (This trick is not suitable for sum aggregation.)
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
        """
        Element‑wise combine the aggregated backup with the previous values via `min` or `max`, depending on `aggr`.
        """
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

    atoi_map = {
        "t": 0,
        "q": 1,
        "p": 2,
    }
    from plangolin.algorithms import mdp_graph_as_pyg_data

    pyg_graph = mdp_graph_as_pyg_data(graph)
    pyg_graph.atoms_per_state = [["p"], ["q"], ["t"]]
    mp_module = OptimalAtomValuesMP(atom_to_index_map=atoi_map, aggr="min")
    mp_module(pyg_graph)
    final_distances = pyg_graph[OptimalAtomValuesMP.default_attr_name]
    expected = torch.tensor([[2, 1, 0], [1, 0, 2], [0, 2, 1]], dtype=torch.float)
    assert final_distances.shape == expected.shape
    assert torch.allclose(final_distances, expected)
