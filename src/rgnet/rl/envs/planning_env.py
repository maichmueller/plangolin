from __future__ import annotations

import abc
import dataclasses
from itertools import cycle
from typing import Generic, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.base import CompatibleType
from torch.nn import Parameter
from torchrl.data import Bounded, Categorical, Composite, NonTensor
from torchrl.envs import EnvBase

import xmimir as xmi
from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, as_non_tensor_stack
from rgnet.rl.reward import RewardFunction

InstanceType = TypeVar("InstanceType")


class InstanceReplacementStrategy(metaclass=abc.ABCMeta):
    """Strategies that determine which instance to use next.
    This is called each time a batch entry is reset, either because a done signal was encountered or
    .reset() was called on the environment to create a new rollout."""

    def __init__(self, all_instances: List[InstanceType]):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, index: int) -> InstanceType:
        """Gets the index of the batch_entry which is being reset as input."""
        pass


class RoundRobinReplacement(InstanceReplacementStrategy):
    def __init__(self, all_instances: List[InstanceType]):
        super().__init__(all_instances)
        self._next_active_iterator = cycle(all_instances)

    def __call__(self, index: int) -> InstanceType:
        return next(self._next_active_iterator)


class PlanningEnvironment(EnvBase, Generic[InstanceType], metaclass=abc.ABCMeta):
    """
    This is the parent environment for all planning environments. Children of this class
    can implement the transition and reset dynamics through the abstract functions.
    In particular using this architecture allows the use of StateSpace's as well as
    SuccessorGenerators.
    The environment can handle multiple different instances by keeping a list of currently
    active instances.
    If more state-spaces are provided than the batch-size they will be circulated in a
    round-robin manner.
    Note that this means that during a rollout states of a single batch can stem from
    different instances.

    There will always be at least one possible transition, if transitions_for returns
    an empty list the environment will add a transition that leads to the same state.
    After such a transition the episode will be terminated.
    """

    batch_locked: bool = True
    # Default rewards
    # The reward precedence is default reward < dead end reward < goal reward.
    # In order to avoid unnecessary dead-end trajectories we give a custom reward
    # instead which should be equal to an infinite trajectory.
    default_dead_end_reward: float = -10.0  # this should be set to 1 / (1 - gamma)
    default_goal_reward: float = 0.0
    default_reward: float = -1.0

    @dataclasses.dataclass(frozen=True)
    class AcceptedKeys:
        state: NestedKey = "state"
        goals: NestedKey = "goals"
        transitions: NestedKey = "transitions"
        instance: NestedKey = "instance"
        action: NestedKey = "action"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        truncated: NestedKey = "truncated"

    default_keys = AcceptedKeys()

    def __init__(
        self,
        all_instances: List[InstanceType],
        reward_function: RewardFunction,
        batch_size: torch.Size | int | Iterable[int],
        seed: Optional[int] = None,
        device: str = "cpu",
        keys: AcceptedKeys = default_keys,
        reset: bool = False,
    ):
        batch_size = PlanningEnvironment.assert_1d_batch(batch_size)
        super().__init__(
            device=device,
            batch_size=batch_size,
        )

        self._keys = keys
        self._rng: torch.Generator
        self.set_seed(seed)

        batch_size = self.batch_size[0]

        self._all_instances: List[InstanceType] = all_instances

        # will be initialized by the first call to _reset
        self._active_instances: List[InstanceType] = [None] * batch_size
        # We iterate over all instances with a cyclic iterator.
        self._instance_replacement_strategy = RoundRobinReplacement(all_instances)

        # We return the unit cost of one (reward=-1) for every action.
        # We use Parameter such that device changes will move this tensor too.
        self._default_reward_tensor = Parameter(
            torch.full(
                size=self.batch_size,
                fill_value=self.default_reward,
                dtype=torch.float,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.reward_function = reward_function
        self._make_spec()
        if reset:
            self.reset()

    @property
    def keys(self):
        return self._keys

    @property
    def active_instances(self):
        return self._active_instances

    @staticmethod
    def assert_1d_batch(batch_size: torch.Size | int | Iterable[int]):
        # We allways require a batch of one which means you will need to access
        # tensordict['action'][0] to get the action even though the batch-size is 1.
        batch_size = (
            torch.Size((batch_size,))
            if isinstance(batch_size, int)
            else torch.Size(batch_size)
        )
        assert len(batch_size) == 1, "Only 1D batches are implemented"
        return batch_size

    def _make_spec(self):
        """Configure environment specification."""

        batch_size = self.batch_size
        self.observation_spec = Composite(
            **{
                # a pymimir.State object
                self._keys.state: NonTensor(shape=batch_size),
                # a List[pymimir.Transition] might be empty
                self._keys.transitions: NonTensor(shape=batch_size),
                # a pymimir.LiteralList object
                self._keys.goals: NonTensor(shape=batch_size),
                # an instance object, e.g. a StateSpace or a SuccessorGenerator
                # The state, transitions and goals are all related to this instance.
                self._keys.instance: NonTensor(shape=batch_size),
            },
            shape=batch_size,
        )
        # Defines what else the step function requires beside the "action" entry
        self.state_spec = Composite(shape=batch_size)  # a.k.a. void
        # For states without outgoing transitions the action will be None
        self.action_spec = NonTensor(shape=batch_size)  # Optional[pymimir.State]
        self.reward_spec: Bounded = Bounded(
            low=-1.0,
            high=1.0,
            dtype=torch.float32,
            shape=torch.Size((*batch_size, 1)),
        )
        self.done_spec = Composite(
            **{
                # a boolean tensor indicating whether the episode is done
                self.keys.done: Categorical(
                    n=2, dtype=torch.bool, shape=torch.Size((*batch_size, 1))
                ),
                self.keys.terminated: Categorical(
                    n=2, dtype=torch.bool, shape=torch.Size((*batch_size, 1))
                ),
                # We don't set truncated, but can be set in rollout
                self.keys.truncated: Categorical(
                    n=2, dtype=torch.bool, shape=torch.Size((*batch_size, 1))
                ),
            },
            shape=torch.Size(batch_size),
        )

    @abc.abstractmethod
    def transitions_for(
        self, active_instance: InstanceType, state: xmi.XState
    ) -> List[xmi.XTransition]:
        """
        Return all transitions that can be taken from the state.
        :param active_instance: The instance the state is part of.
        :param state: The state from which the transitions can be taken.
        """
        pass

    @abc.abstractmethod
    def initial_for(
        self, active_instance: InstanceType
    ) -> Tuple[xmi.XState, List[xmi.XLiteral]]:
        """
        :param active_instance: The instance after this batch entry was reset.
        :return: the new initial state and goals for the newly reset instance.
        """
        pass

    @abc.abstractmethod
    def is_goal(self, active_instance: InstanceType, state: xmi.XState) -> bool:
        """
        :param active_instance: The instance the state is part of.
        :param state: The state which should be checked against the goal.
        :return: Whether the state is a goal-state.
        """
        pass

    def create_td(
        self, source: TensorDictBase | dict[str, CompatibleType] | None = None
    ) -> TensorDict:
        """Generate an empty tensordict with the correct device and batch-size.."""
        return TensorDict(source or {}, batch_size=self.batch_size, device=self.device)

    @staticmethod
    def is_dead_end_transition(transition: xmi.XTransition) -> bool:
        return transition.action is None

    def get_reward_and_done(
        self,
        transitions: Sequence[xmi.XTransition],
        instances: Sequence[InstanceType] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the reward and done signal for the current state and actions taken.
        The method is typically called from step, therefore current_states refers to the states
        before the actions are taken.
        The batch dimension can be over the environment batch size or the time.
        :param transitions: The actions taken by the agent.
        :param instances the instances from which actions and current states stem from.
            Requires len(instances) == len(transitions).
            This parameter can be used to get the rewards and done signals after a rollout was already finished.
            Defaults to self._active_instances.

        :return A tuple containing the rewards and done signal for the actions
        """
        instances = instances or self._active_instances
        if len(transitions) != len(instances):
            if len(instances) != 1:
                raise ValueError(
                    f"The batch dimension of transitions and instances must be compatible.\n"
                    f"Got {[len(transitions)]} and {[len(instances)]}."
                )
            else:
                instances = instances * len(transitions)
        done = []
        labels = []
        for transition, active_instance in zip(transitions, instances):
            if self.is_goal(active_instance, transition.source):
                done.append(True)
                labels.append(xmi.StateLabel.goal)
            elif self.is_dead_end_transition(transition):
                done.append(True)
                labels.append(xmi.StateLabel.deadend)
            else:
                done.append(False)
                labels.append(xmi.StateLabel.default)
        rewards = torch.tensor(
            self.reward_function(transitions, labels),
            dtype=torch.float,
            device=self.device,
        )
        return rewards, torch.tensor(done, dtype=torch.bool, device=self.device)

    def get_applicable_transitions(
        self, states: List[xmi.XState]
    ) -> List[List[xmi.XTransition]]:
        """
        For dead-end states or goal-states we add an artificial self-transition without real action (None).

        Note: we can never have empty transitions, since value-learning would not work for terminating-states
        (goal/deadend) otherwise (since these states would never appear as decision state in a typical rollout of the
        environment.
        Hence, our environment will always add a self-transition from terminating states (goal/deadends) to themselves
        with a None action to mark the termination.
        """
        return [
            (
                not self.is_goal(instance, state)
                and self.transitions_for(instance, state)
            )
            or [xmi.XTransition.make_hollow(state, None, state)]
            for (instance, state) in zip(self._active_instances, states)
        ]

    def rand_action(
        self, tensordict: Optional[TensorDictBase] = None
    ) -> TensorDictBase:
        """Generate a random action.

        Args:
            tensordict (Optional[TensorDictBase], optional): Input TensorDict. Defaults to None.

        Returns:
            TensorDictBase: Output TensorDict.
        """
        if tensordict is None:
            tensordict = self.reset()  # we need transitions to choose from

        PlanningEnvironment.assert_1d_batch(tensordict.batch_size)

        # using [] should automatically trigger .tolist for NonTensorData/Stack
        batched_transitions = tensordict[self._keys.transitions]
        assert isinstance(batched_transitions, List)

        tensordict[self._keys.action] = as_non_tensor_stack(
            [
                transitions[
                    torch.randint(0, len(transitions), (1,), generator=self._rng)
                ]
                for transitions in batched_transitions
            ]
        )
        return tensordict

    def _apply_transition_or_stay(
        self,
        idx: int,
        transition: Optional[xmi.XTransition],
        current_states: List[xmi.XState],
    ):
        if transition is not None:
            return transition.target
        else:
            # Check that there were no transitions available.
            state = current_states[idx]
            assert (
                len(self.transitions_for(self._active_instances[idx], state)) == 0
            ), "No transitions were given for state with available transitions."
        return state

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Perform a step in the environment.
        Apply the action, compute a new state, render pixels and determine reward, termination and valid next actions.
        The trajectory is done after a goal state is visited!
        NOTE that EnvBase.step() already checks that the batch_size matches.
        :param td (TensorDict): TensorDict with state and action.
        :returns TensorDict: Output TensorDict.
        """

        current_states: List[xmi.XState] = tensordict[self._keys.state]

        actions: List[Optional[xmi.XTransition]] = tensordict[self._keys.action]
        assert isinstance(actions, list)  # batch of chosen-transitions

        # Apply the transition or stay in the current state if none are available.
        next_states = [
            self._apply_transition_or_stay(i, transition, current_states)
            for i, transition in enumerate(actions)
        ]

        applicable_transitions = self.get_applicable_transitions(next_states)
        # We terminate if either we came from a goal or from a dead end.
        reward, done = self.get_reward_and_done(actions)
        assert reward.shape == done.shape

        return self.create_td(
            {
                self._keys.state: as_non_tensor_stack(next_states),
                self._keys.transitions: as_non_tensor_stack(applicable_transitions),
                self._keys.goals: tensordict.get(self._keys.goals),
                self._keys.instance: as_non_tensor_stack(self._active_instances),
                self._keys.reward: reward,
                self._keys.done: done,
                self._keys.terminated: done,
                # issue: https://github.com/pytorch/rl/issues/2291
            }
        )

    def _set_seed(self, seed: Optional[int]):
        """Initialize random number generator with given seed.
        :param seed (Optional[int]): Seed.
        """
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self._rng = torch.manual_seed(seed)

    def _reset(
        self,
        tensordict: Optional[TensorDict],
        states: List[xmi.State] | NonTensorWrapper | None = None,
        **kwargs,
    ) -> TensorDict:
        batch_size = self.batch_size[0]

        if (
            tensordict is not None
            and (partial_reset := tensordict.get("_reset", None)) is not None
        ):
            if states is not None:
                raise ValueError("Can't provide initial states during a partial reset.")
            # List of indices where a reset has to occur
            if partial_reset.ndim == 2:
                partial_reset = partial_reset.squeeze(
                    -1
                )  # unpack torch.tensor([[True]])
            indices_to_reset: List = partial_reset.nonzero().squeeze(-1).tolist()
        else:
            indices_to_reset = list(range(batch_size))

        for index in indices_to_reset:
            self._active_instances[index] = self._instance_replacement_strategy(index)

        initial_states, initial_goals = zip(
            *[self.initial_for(instance) for instance in self._active_instances]
        )
        if states is not None:
            assert len(states) == self.batch_size[0]
            initial_states = states

        initial_transitions = self.get_applicable_transitions(initial_states)

        out = self.create_td(
            {
                self._keys.state: as_non_tensor_stack(initial_states),
                self._keys.transitions: as_non_tensor_stack(initial_transitions),
                self._keys.goals: as_non_tensor_stack(initial_goals),
                self._keys.instance: as_non_tensor_stack(self._active_instances),
            }
        )

        return out

    def make_replacement_strategy(
        self, strategy_class: Type[InstanceReplacementStrategy], **kwargs
    ):
        self._instance_replacement_strategy = strategy_class(
            all_instances=self._all_instances, **kwargs
        )
