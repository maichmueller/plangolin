import math
import warnings
from functools import singledispatch
from typing import Dict, List, Set

import networkx as nx
import torch
import torch_geometric as pyg
from tensordict import NestedKey
from torch import Tensor
from torchrl.modules import ValueOperator

import xmimir as xmi
from plangolin.rl.envs import ExpandedStateSpaceEnv
from plangolin.rl.reward import UnitReward
from plangolin.rl.reward.uniform_reward import FlatReward
from plangolin.utils.misc import NonTensorWrapper, broadcastable, tolist
from xmimir import XState, XStateSpace

from .mdp_to_pyg import mdp_graph_as_pyg_data
from .policy_evaluation_mp import OptimalPolicyMP, ValueIterationMP


@singledispatch
def optimal_policy(
    env: ExpandedStateSpaceEnv, optimal_values: list[Tensor] | Tensor | None = None
) -> Dict[int, Set[int]]:
    if type(env.reward_function) is UnitReward:
        reward_func: UnitReward = env.reward_function
        if type(env.active_instances[0]) is xmi.XStateSpace:
            # the underlying state space is a pure state space (no macro transitions)
            if math.isclose(reward_func.regular_reward, -1.0, abs_tol=1e-8):
                return optimal_policy(env.active_instances[0])
    if optimal_values is not None:
        return optimal_policy(env.active_instances[0], optimal_values=optimal_values)
    else:
        return optimal_policy(
            env.active_instances[0], optimal_values=bellman_optimal_values(env)
        )


@optimal_policy.register
def _(
    space: XStateSpace, optimal_values: list[Tensor] | Tensor | None = None
) -> Dict[int, Set[int]]:
    """
    Computes the optimal policy for the given state space.

    If `optimal_values` is provided, it is used to determine the optimal actions.
    Assumptions:
        - optimal_values[i] is the value of the state for which state.index == i
        - optimal_values[i][j] is the value of the transition space.forward_transitions(state)[j]
    """
    match optimal_values:
        case None:
            # optimal_values is None, so we use the goal distance --> argmin
            def optimal_successors(state: XState) -> Tensor:
                return torch.tensor(
                    [
                        space.goal_distance(t.target)
                        for t in space.forward_transitions(state)
                    ],
                    dtype=torch.float,
                ).argmin()

        case list():
            # optimal_values represents rewards, so we need to use argmax
            assert isinstance(optimal_values[0], Tensor)

            def optimal_successors(state: XState) -> Tensor:
                return optimal_values[state.index].argmax()

        case Tensor():
            # optimal_values represents rewards, so we need to use argmax
            # we assume here that the tensor is a 1D tensor with the same length as the number of states,
            # in which optimal_values[i] is the optimal value of the state for which state.index == i
            assert broadcastable(optimal_values.shape, (len(space),))
            optimal_values = optimal_values.view(len(space))

            def optimal_successors(state: XState) -> Tensor:
                return torch.tensor(
                    [
                        optimal_values[t.target.index]
                        for t in space.forward_transitions(state)
                    ],
                    dtype=torch.float,
                ).argmax()

        case _:
            raise TypeError(
                f"optimal_values must be None, a list of Tensors or a Tensor, but got {type(optimal_values)}"
            )

    # index of state to set of indices of optimal actions
    # optimal[i] = {j},  0 <= j < len(space.forward_transitions(space[i]))
    optimal: Dict[int, Set[int]] = dict()
    for state in space:
        if space.is_deadend(state):
            optimal[state.index] = set()
            continue
        best_actions: Set[int] = set(optimal_successors(state).view(-1).tolist())
        optimal[state.index] = best_actions
    return optimal


@optimal_policy.register
def _(
    space_graph: nx.DiGraph, optimal_values: list[Tensor] | Tensor | None = None
) -> Dict[int, Set[int]]:
    """
    Computes the optimal policy for the given state MDP graph.

    If `optimal_values` is provided, it is used to determine the optimal actions.
    Assumptions:
        - optimal_values[i] is the value of the state for which state.index == i
        - optimal_values[i][j] is the value of the transition space.forward_transitions(state)[j]
    """
    if optimal_values is not None:
        # optimal_values represents rewards, so we need to use argmax

        def optimal_successors(node) -> Tensor:
            return optimal_values[node].argmax()

    else:
        # optimal_values is None, so we use the goal distance --> argmin

        def optimal_successors(node) -> Tensor:
            return torch.tensor(
                [
                    space_graph.nodes[target]["dist"]
                    for _, target in space_graph.edges(node)
                ],
                dtype=torch.float,
            ).argmin()

    # index of state to set of indices of optimal actions
    # optimal[i] = {j},  0 <= j < len(space.forward_transitions(space[i]))
    optimal: Dict[int, Set[int]] = dict()
    for node, data in space_graph.nodes.items():
        if data["ntype"] == "deadend":
            optimal[node] = set()
            continue
        best_actions: Set[int] = set(optimal_successors(node).view(-1).tolist())
        optimal[node] = best_actions
    return optimal


@optimal_policy.register
def _(space_data: pyg.data.Data, optimal_values: list[Tensor] | Tensor | None = None):
    if isinstance(optimal_values, torch.Tensor):
        assert broadcastable(optimal_values.shape, (space_data.num_nodes,))
        setattr(
            space_data,
            ValueIterationMP.default_attr_name,
            optimal_values.view(space_data.num_nodes),
        )
    elif isinstance(optimal_values, list):
        warnings.warn("Ignoring optimal_values given as list, recomputing them.")

    return OptimalPolicyMP(gamma=space_data.gamma)(space_data)


def discounted_value(distance_to_goal, gamma):
    return -(1 - gamma**distance_to_goal) / (1 - gamma)


@singledispatch
def bellman_optimal_values(env: ExpandedStateSpaceEnv, **kwargs) -> torch.Tensor:
    reward_type = type(env.reward_function)
    if (
        reward_type is UnitReward or reward_type is FlatReward
    ):  # 'is' to ensure no derived class
        reward_func: UnitReward = env.reward_function
        if (
            type(env.active_instances[0]) is xmi.XStateSpace
            or reward_type is FlatReward
        ):
            # the underlying state space is a pure state space (no macro transitions) or
            # we are crediting all transitions with the same reward anyway
            if math.isclose(reward_func.regular_reward, -1.0, abs_tol=1e-8):
                return bellman_optimal_values(
                    env.active_instances[0], gamma=reward_func.gamma
                )
        kwargs["gamma"] = reward_func.gamma
    graph = env.to_mdp_graph(0)
    return bellman_optimal_values(graph, **kwargs)


@bellman_optimal_values.register
def _(space: XStateSpace, gamma: float) -> torch.Tensor:
    return torch.tensor(
        [discounted_value(space.goal_distance(s), gamma=gamma) for s in space],
        dtype=torch.float,
    )


@bellman_optimal_values.register
def _(env: nx.DiGraph, **kwargs) -> torch.Tensor:
    return bellman_optimal_values(mdp_graph_as_pyg_data(env), **kwargs)


@bellman_optimal_values.register
def _(env: pyg.data.Data, **kwargs) -> torch.Tensor:
    """
    Computes the Bellman optimal values for the given environment using value iteration.
    """
    if hasattr(env, "gamma"):
        kwargs["gamma"] = kwargs.get("gamma", env.gamma)
    return ValueIterationMP(**kwargs)(env)


class OptimalValueFunction(torch.nn.Module):
    """Don't predict the value target just use the discounted distance to goal"""

    def __init__(self, optimal_values: Dict[xmi.XState, float], device: torch.device):
        super().__init__()
        self.optimal_values = optimal_values
        self.device = device

    def __call__(
        self, batched_states: List[xmi.XState] | NonTensorWrapper
    ) -> torch.Tensor:
        batched_states = tolist(batched_states)
        return torch.stack(
            [
                torch.tensor(
                    [self.optimal_values[state] for state in states],
                    dtype=torch.float,
                    device=self.device,
                ).view(-1, 1)
                for states in batched_states
            ]
        )

    def as_td_module(
        self, state_key: NestedKey, state_value_key: NestedKey
    ) -> ValueOperator:
        return ValueOperator(
            module=self, in_keys=[state_key], out_keys=[state_value_key]
        )
