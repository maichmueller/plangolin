import abc
import itertools
from collections import defaultdict
from copy import copy
from functools import cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import networkx as nx
import torch
import torch_geometric as pyg
from torch import Tensor
from torchrl.data.utils import DEVICE_TYPING

import xmimir as xmi
from rgnet.logging_setup import tqdm
from rgnet.rl.reward import RewardFunction, UnitReward
from rgnet.utils.misc import copy_return
from rgnet.utils.reshape import unsqueeze_right_like
from xmimir import StateType, XLiteral, XProblem, XState, XStateSpace, XTransition

from .planning_env import InstanceReplacementStrategy, PlanningEnvironment
from .successor_env import SuccessorEnvironment


class ResetStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, space: xmi.XStateSpace) -> xmi.XState:
        pass


class InitialStateReset(ResetStrategy):
    def __call__(self, space: xmi.XStateSpace):
        return space.initial_state


class UniformRandomReset(ResetStrategy):
    def __init__(self, generator: Optional[torch.Generator] = None):
        self.generator = generator

    def __call__(self, space: xmi.XStateSpace):
        return space[
            torch.randint(0, len(space), size=(1,), generator=self.generator).item()
        ]


class WeightedRandomReset(InstanceReplacementStrategy):
    def __init__(
        self,
        all_instances: List[xmi.XStateSpace],
        generator: Optional[torch.Generator] = None,
    ):
        total_states = sum(len(space) for space in all_instances)
        self.weights = torch.tensor(
            [len(space) / total_states for space in all_instances]
        )
        self.all_instances = all_instances
        self.generator = generator
        super().__init__(all_instances)

    def __call__(self, index: int) -> xmi.XStateSpace:
        index = torch.multinomial(
            self.weights, 1, replacement=True, generator=self.generator
        ).item()
        return self.all_instances[index]


class IteratingReset(ResetStrategy):
    """
    Iterate over each state in the space in cycles.
    For each space separate counters are stores.
    """

    def __init__(self):
        super().__init__()
        self.idx_per_space: Dict[xmi.XStateSpace, int] = defaultdict(int)

    def __call__(self, space: xmi.XStateSpace) -> xmi.XState:
        idx = self.idx_per_space[space]
        self.idx_per_space[space] = (idx + 1) % len(space)
        return space.get_state(idx)


def _serialized_action_info(transition):
    if isinstance(transition.action, Sequence):
        action_data = (
            "\n".join(str(a) for a in transition.action),
            tuple(a.index for a in transition.action),
        )
    elif transition.action is None:
        action_data = None
    else:
        action_data = str(transition.action), transition.action.index
    return action_data


def _state_type(instance: XStateSpace, state: XState, goal: tuple[XLiteral, ...] = ()):
    if state.is_goal(goal):
        return StateType.GOAL
    elif instance.initial_state == state:
        return StateType.INITIAL
    elif instance.is_deadend(state):
        return StateType.DEADEND
    else:
        return StateType.DEFAULT


class MultiInstanceStateSpaceEnv(PlanningEnvironment[xmi.XStateSpace]):
    def __init__(
        self,
        spaces: List[xmi.XStateSpace],
        reset_strategy: ResetStrategy = InitialStateReset(),
        batch_size: torch.Size | int | Iterable[int] = 1,
        seed: Optional[int] = None,
        device: DEVICE_TYPING = "cpu",
        keys: PlanningEnvironment.AcceptedKeys = PlanningEnvironment.default_keys,
        reward_function: RewardFunction = UnitReward(gamma=0.9),
        **kwargs,
    ):
        self.reset_strategy = reset_strategy
        super().__init__(
            all_instances=spaces,
            batch_size=batch_size,
            seed=seed,
            device=device,
            keys=keys,
            reward_function=reward_function,
            **kwargs,
        )

    @property
    def spaces(self):
        return self._all_instances

    def transitions_for(
        self, active_instance: xmi.XStateSpace, state: xmi.XState
    ) -> List[xmi.XTransition]:
        return list(active_instance.forward_transitions(state))

    def initial_for(
        self, active_instance: xmi.XStateSpace
    ) -> Tuple[xmi.XState, tuple[xmi.XLiteral, ...]]:
        return (
            self.reset_strategy(active_instance),
            active_instance.problem.goal(),
        )

    def is_goal(self, active_instance: xmi.XStateSpace, state: XState) -> bool:
        return active_instance.is_goal(state)

    @copy_return
    @cache
    def to_pyg_data(
        self,
        instance_index: int,
        transition_probabilities: (
            tuple[Tensor, ...]
            | Callable[[XState, Sequence[XTransition]], Tensor]
            | None
        ) = None,
        natural_transitions: bool = False,
        problems: tuple[XProblem, ...] | None = None,
        # progressbar: bool = False,
        progressbar: bool = True,
    ) -> pyg.data.Data:
        r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance_list.

        :params:
            transition_probabilities (tuple[Tensor] | Callable[[XState, Sequence[XTransition]], Tensor] | None):
                The transition probabilities for each state. If None, uniform transition probabilities are used.

        """
        space = self._all_instances[instance_index]

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
            assert len(transition_probabilities) == len(space)

            def transition_probs(s: XState, ts: Sequence[XTransition]):
                return transition_probabilities[s.index]

        else:
            transition_probs = transition_probabilities
        device = "cpu"

        if natural_transitions:
            transitions = tuple(
                tuple(space.forward_transitions(state)) for state in space
            )
        else:
            transitions = self.get_applicable_transitions(
                space, instances=itertools.repeat(space)
            )

        edge_index = torch.empty(
            (
                2,
                sum(len(t) for t in transitions),
            ),
            dtype=torch.long,
            device=device,
        )
        group_edge_attrs = ["reward", "probs", "done", "idx"]

        data_dict: Dict[str, Any] = defaultdict(list)
        data_dict["edge_index"] = edge_index
        data_dict["gamma"] = self.reward_function.gamma

        if problems:
            successor_gen = space.successor_generator
            successor_gens = []
            for problem in problems:
                successor_gen_copy = copy(successor_gen)
                successor_gen_copy.problem = problem
                successor_gens.append(successor_gen_copy)
            aux_env = SuccessorEnvironment(
                successor_gens, reward_function=self.reward_function, batch_size=1
            )
            instance_iter = itertools.repeat(successor_gens)
            problem_iter = itertools.repeat(problems)
        else:
            aux_env = None
            problem_iter = itertools.repeat([space.problem])
            instance_iter = itertools.repeat([space])
        transition_index = itertools.count(0)
        nr_instances = len(next(instance_iter))
        goal_rewards = torch.empty(
            (len(space), nr_instances), dtype=torch.float, device=device
        )
        query_env = aux_env or self
        iters = zip(space, instance_iter, problem_iter, transitions)
        for state, instance_list, problem_list, state_transitions in (
            tqdm(
                iters,
                total=len(space),
                desc=f"Env-to-PyG ({len(space)} states, {space.total_transition_count} transitions",
            )
            if progressbar
            else iters
        ):
            attr_shape = (-1, nr_instances)
            rewards, done = zip(
                *(
                    map(
                        lambda t: t.to(device).view(1, -1),
                        query_env.get_reward_and_done(
                            state_transitions,
                            instances=[instance] * len(state_transitions),
                        ),
                    )
                    for instance in instance_list
                )
            )
            rewards, done = map(lambda d: torch.cat(d, dim=0).T, (rewards, done))
            t_probs: torch.Tensor = (
                unsqueeze_right_like(
                    transition_probs(state, state_transitions).float(),
                    rewards,
                )
                .to(device)
                .expand_as(rewards)
            )
            transition_indices = list(
                next(transition_index) for _ in range(len(state_transitions))
            )
            dist = torch.tensor(
                [
                    (
                        instance.goal_distance(state)
                        if isinstance(instance, XStateSpace)
                        else torch.nan
                    )
                    for instance in instance_list
                ],
                dtype=torch.float,
                device=device,
            ).view(attr_shape)
            node_type = torch.tensor(
                [
                    (
                        _state_type(instance, state, problem.goal()).value
                        if isinstance(instance, XStateSpace)
                        else -1
                    )
                    for instance, problem in zip(instance_list, problem_list)
                ],
                dtype=torch.int,
                device=device,
            ).view(attr_shape)
            is_goal_state = torch.tensor(
                [query_env.is_goal(instance, state) for instance in instance_list],
                dtype=torch.bool,
                device=device,
            ).view(attr_shape)
            goal_rewards[state.index] = torch.where(
                is_goal_state, rewards.max(dim=0)[0], -torch.inf
            ).view(attr_shape)
            data_dict["dist"].append(dist)
            data_dict["ntype"].append(node_type)
            data_dict["goals"].append(is_goal_state)
            data_dict["reward"].append(rewards)
            data_dict["done"].append(done)
            data_dict["probs"].append(t_probs)
            data_dict["action"].extend(
                _serialized_action_info(t) for t in state_transitions
            )
            data_dict["idx"].append(
                unsqueeze_right_like(
                    torch.tensor(transition_indices, dtype=torch.int), rewards
                )
                .to(device)
                .expand_as(rewards)
            )

            edge_index[0, transition_indices] = state.index
            edge_index[1, transition_indices] = torch.tensor(
                [t.target.index for t in state_transitions],
                dtype=torch.long,
                device=device,
            )
        for key, value in data_dict.items():
            if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
                data_dict[key] = torch.cat(value, dim=0).squeeze(-1)
            else:
                try:
                    data_dict[key] = torch.as_tensor(value, device=device)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception:
                    pass

        data = pyg.data.Data.from_dict(data_dict)
        if group_edge_attrs is not None:
            xs = []
            for key in group_edge_attrs:
                x = data[key]
                x = x.unsqueeze(-1) if x.dim() <= 2 else x
                xs.append(x)
                del data[key]
            data.edge_attr = torch.cat(xs, dim=-1)

        # goal states have the value of their reward (typically 0, but could be arbitrary);
        # the rest is initialized to 0.
        data.x = torch.where(
            unsqueeze_right_like(data.goals, goal_rewards),
            goal_rewards,
            torch.zeros((len(space), nr_instances), device=device),
        )
        data.num_nodes = len(space)
        return data

    def traverse(
        self,
        index: int,
        natural_transitions: bool = False,
        with_rewards: bool = False,
    ) -> tuple[
        Sequence["XState"],
        Sequence[Sequence["XTransition"]],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        space = self.spaces[index]
        if natural_transitions:
            transitions = list(
                tuple(space.forward_transitions(state)) for state in space
            )
        else:
            transitions = self.get_applicable_transitions(
                space, instances=[space] * len(space)
            )
        if with_rewards:
            rewards, done = self.get_reward_and_done_multi(
                transitions, instances=[[space] * len(ts) for ts in transitions]
            )
        else:
            rewards = done = None
        return space, transitions, rewards, done

    def traverse_iter(
        self,
        index: int,
        natural_transitions: bool = False,
        with_rewards: bool = False,
    ) -> Generator[
        tuple[XState, Sequence[XTransition], Optional[Tensor], Optional[Tensor]],
        None,
        None,
    ]:
        space = self.spaces[index]
        if natural_transitions:
            transitions_iter = (
                tuple(space.forward_transitions(state)) for state in space
            )
        else:
            transitions_iter = (
                self.get_applicable_transitions(state, instances=[space])
                for state in space
            )
        if with_rewards:

            def rewards_done_generator() -> (
                Generator[
                    tuple[Tensor | None, Tensor | None], Sequence[XTransition], None
                ]
            ):
                transitions = yield None, None
                while True:
                    transitions = yield self.get_reward_and_done(
                        transitions, instances=[space] * len(transitions)
                    )

        else:

            def rewards_done_generator() -> (
                Generator[tuple[None, None], Sequence[XTransition], None]
            ):
                yield None, None

        def _traverse_generator():
            rewards_done_gen = rewards_done_generator()
            next(rewards_done_gen)
            for state, transitions in zip(space, transitions_iter):
                rewards, done = rewards_done_gen.send(transitions)
                yield state, transitions, rewards, done

        return _traverse_generator()

    @cache
    def to_mdp_graph(
        self,
        instance_index: int,
        transition_probabilities: (
            tuple[torch.Tensor, ...]
            | Callable[[XState, Sequence[XTransition]], Sequence[float]]
            | None
        ) = None,
        serializable: bool = False,
        natural_transitions: bool = False,
    ) -> nx.MultiDiGraph:
        r"""
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
        instance_index: int,
            the index of the state space to convert.
        transition_probabilities: tuple[
                tuple[torch.Tensor] | Callable[[XState, Sequence[XTransition]], Sequence[float]],
                ...
            ]
            | None,
               The transition probabilities for each state. If None, uniform transition probabilities are used.
        serializable: bool,
                If True, only serializable (pickleable) attributes are used from the state space.
                This is a lossy encoding.
        use_space_directly: bool
               If True, the state space is used directly.
               Otherwise, the environment is used to traverse the space and generate the graph.
               This is significantly faster for large state spaces, since default .traverse() functions
               merely replicate the state space as a tensordict.
        natural_transitions: bool,
            If True, the natural transitions of the state space are used.
            Otherwise, the applicable transitions are used.
            This preserves the structure of the state space. Has no effect if use_space_directly is False.
        """

        if serializable:
            # If serializable is True, we use the state index as nodes.
            # Otherwise, the XState itself is going to form the nodes.

            def state_node(state: XState):
                return state.index

            def emplace_node(
                graph: nx.MultiDiGraph, instance: XStateSpace, state: XState
            ):
                atom_str = ", ".join(map(str, state.atoms(with_statics=False)))
                graph.add_node(
                    state.index,
                    ntype=_state_type(instance, state),
                    atoms=f"[{atom_str}]",
                    dist=instance.goal_distance(state),
                )

        else:

            def state_node(state: XState):
                return state

            def emplace_node(
                graph: nx.MultiDiGraph, instance: XStateSpace, state: XState
            ):
                graph.add_node(
                    state,
                    ntype=_state_type(instance, state),
                    dist=instance.goal_distance(state),
                )

        mdp_graph = nx.MultiDiGraph()

        states, transitions, rewards, done = self.traverse(
            instance_index, natural_transitions, with_rewards=True
        )
        space = self.spaces[instance_index]

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
            is_sequence_of_sequences_of_floats = (
                len(transition_probabilities)
                == len(space)  # the outermost sequence has #entries == #states
                and (
                    isinstance(seq := transition_probabilities[0], Sequence)
                    or (isinstance(seq, torch.Tensor) and seq.dtype == torch.float)
                )
                # the inner sequence/tensor has #entries == #transitions
                and (
                    isinstance(value := transition_probabilities[0][0], float)
                    or (isinstance(value, torch.Tensor) and value.ndim == 0)
                )  # innermost is sequence/tensor over floats --> we have only data for a single state space
            )
            assert is_sequence_of_sequences_of_floats, (
                "Given a sequence type, "
                "we expect a sequence of sequences of floats or a sequence of tensors over floats."
            )
            assert len(transition_probabilities) == len(states)

            def transition_probs(s: XState, ts: Sequence[XTransition]):
                return transition_probabilities[s.index]

        else:
            assert callable(transition_probabilities), (
                "Given transition probabilities are neither a sequence of sequences of floats nor a callable."
                "Please provide a sequence of sequences of floats or a callable."
            )
            transition_probs = transition_probabilities

        for state in states:
            emplace_node(mdp_graph, space, state)

        running_transition_idx = itertools.count(0)
        for state, state_transitions, state_rewards, state_done in zip(
            states, transitions, rewards, done
        ):
            t_probs: Sequence[float] = transition_probs(state, state_transitions)
            for t_idx, t in enumerate(state_transitions):
                action_data = _serialized_action_info(t)
                mdp_graph.add_edge(
                    state_node(t.source),
                    state_node(t.target),
                    action=action_data,
                    reward=state_rewards[t_idx],
                    done=state_done[t_idx],
                    probs=t_probs[t_idx],
                    idx=next(running_transition_idx),
                )
        mdp_graph.graph["gamma"] = self.reward_function.gamma
        return mdp_graph


class ExpandedStateSpaceEnv(MultiInstanceStateSpaceEnv):
    """
    ExpandedStateSpaceEnv defines the MDP environment from a problem's full state space
     as generated by xmimir.
    """

    def __init__(
        self,
        space: xmi.XStateSpace,
        batch_size: torch.Size | int | Iterable[int] = 1,
        **kwargs,
    ):
        batch_size = PlanningEnvironment.assert_1d_batch(batch_size)
        batch_size_ = batch_size[0]
        super().__init__(spaces=[space] * batch_size_, batch_size=batch_size, **kwargs)


def ensure_loaded(func):
    """
    Decorator to ensure that the environment is loaded before calling the function.
    """

    def wrapper(self, *args, **kwargs):
        if not self._all_loaded:
            self._load_all()
        return func(self, *args, **kwargs)

    return wrapper


class ExpandedStateSpaceEnvLoader:
    """
    A callable wrapper that ensures that the environment is loaded before calling the function.

    Needed for LazyEnvLookup to return a callable instead of an environment, that can also be typechecked correctly.
    """

    def __init__(self, env_callable: Callable[[], ExpandedStateSpaceEnv]):
        self.loader_callable = env_callable

    def __call__(self) -> ExpandedStateSpaceEnv:
        return self.loader_callable()


class LazyEnvLookup(Mapping[Path, ExpandedStateSpaceEnv]):
    """
    A dictionary-like object that loads environments for a given problem path upon __getitem__

    Expected to be instantiated with a list of problem paths and a callable that takes a path and returns an environment.
    """

    def __init__(
        self,
        problems: Iterable[Path],
        env_callable: Callable[[Path], ExpandedStateSpaceEnv],
        loaded_envs: Dict[Path, ExpandedStateSpaceEnv] | None = None,
    ):
        problems = tuple(problems)
        self.problems = {problem: i for i, problem in enumerate(problems)}
        self.envs: list[ExpandedStateSpaceEnv | None] = [None] * len(problems)
        if loaded_envs is not None:
            for problem in problems:
                if problem in loaded_envs:
                    self.envs[self.problems[problem]] = loaded_envs[problem]
        self.env_callable = env_callable
        self._all_loaded = False

    def _load_all(self):
        for problem in self.problems:
            if self.envs[self.problems[problem]] is None:
                self.envs[self.problems[problem]] = self.env_callable(problem)
        self._all_loaded = True

    @ensure_loaded
    def keys(self):
        return self.problems.keys()

    @ensure_loaded
    def values(self):
        return self.envs

    @ensure_loaded
    def items(self):
        return zip(self.problems.keys(), self.envs)

    def __len__(self):
        return len(self.problems)

    @ensure_loaded
    def __iter__(self):
        return self.keys()

    def __call__(
        self, path: Path | str
    ) -> ExpandedStateSpaceEnv | ExpandedStateSpaceEnvLoader:
        return ExpandedStateSpaceEnvLoader(lambda: self[path])

    def __getitem__(self, path: Path | str) -> ExpandedStateSpaceEnv:
        path = Path(path)
        if path not in self.problems:
            raise KeyError(f"Problem {path} not member of the environments to load.")
        env = self.envs[self.problems[path]]
        if env is None:
            env = self.env_callable(path)
            self.envs[self.problems[path]] = env
        self.all_loaded = all(env is not None for env in self.envs)
        return env

    def __setitem__(self, key, value):
        raise NotImplementedError("This dictionary is read-only.")
