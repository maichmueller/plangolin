import abc
import dataclasses
from itertools import cycle
from typing import Generic, List, Optional, Tuple, TypeVar

import pymimir as mi
import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.base import CompatibleType
from torchrl.data import BoundedTensorSpec, CompositeSpec, NonTensorSpec
from torchrl.envs import EnvBase

from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, as_non_tensor_stack

InstanceType = TypeVar("InstanceType")


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
    """

    batch_locked: bool = True

    @dataclasses.dataclass(frozen=True)
    class AcceptedKeys:
        state: NestedKey = "state"
        goals: NestedKey = "goals"
        transitions: NestedKey = "transitions"
        action: NestedKey = "action"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        truncated: NestedKey = "truncated"

    default_keys = AcceptedKeys()

    def __init__(
        self,
        all_instances: List[InstanceType],
        batch_size: torch.Size,
        seed: Optional[int] = None,
        device: str = "cpu",
        keys: AcceptedKeys = default_keys,
    ):
        PlanningEnvironment.assert_1D_batch(batch_size)
        super().__init__(device=device, batch_size=batch_size)

        self._keys = keys
        self._rng: torch.Generator
        self.set_seed(seed)

        batch_size = self.batch_size[0]

        self._all_instances: List[InstanceType] = all_instances

        # will be initialized by the first call to _reset
        self._active_instances: List[InstanceType] = [None] * batch_size
        # We iterate over all instances with a cyclic iterator.
        self._next_active_iterator = cycle(self._all_instances)

        # We return the unit cost of one (reward=-1) for every action.
        self._minus_one_rewards = torch.full(
            size=self.batch_size, fill_value=-1.0, dtype=torch.float
        )
        self._make_spec()

    @property
    def keys(self):
        return self._keys

    @staticmethod
    def assert_1D_batch(batch: torch.Size):
        # We allways require a batch of one which means you will need to access
        # tensordict['action'][0] to get the action even though the batch-size is 1.
        assert len(batch) == 1, "Only 1D batches are implemented"

    def _make_spec(self):
        """Configure environment specification."""

        batch_size = self.batch_size
        self.observation_spec = CompositeSpec(
            **{
                # a pymimir.State object
                self._keys.state: NonTensorSpec(shape=batch_size),
                # a List[pymimir.Transition] might be empty
                self._keys.transitions: NonTensorSpec(shape=batch_size),
                # a pymimir.LiteralList object
                self._keys.goals: NonTensorSpec(shape=batch_size),
            },
            shape=batch_size,
        )
        # Defines what else the step function requires beside the "action" entry
        self.state_spec = CompositeSpec(shape=batch_size)  # a.k.a. void
        # For states without outgoing transitions the action will be None
        self.action_spec = NonTensorSpec(shape=batch_size)  # Optional[pymimir.State]
        self.reward_spec: BoundedTensorSpec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            dtype=torch.float32,
            shape=torch.Size([batch_size[0], 1]),
        )

    @abc.abstractmethod
    def transitions_for(
        self, active_instance: InstanceType, state: mi.State
    ) -> List[mi.Transition]:
        """
        Return all transitions that can be taken from the state.
        :param active_instance: The instance the state is part of.
        :param state: The state from which the transitions can be taken.
        """
        pass

    @abc.abstractmethod
    def initial_for(
        self, active_instance: InstanceType
    ) -> Tuple[mi.State, List[mi.Literal]]:
        """
        :param active_instance: The instance after this batch entry was reset.
        :return: the new initial state and goals for the newly reset instance.
        """
        pass

    @abc.abstractmethod
    def is_goal(self, active_instance: InstanceType, state: mi.State) -> bool:
        """
        :param active_instance: The instance the state is part of.
        :param state: The state which should be checked against the goal.
        :return: Whether the state is a goal-state.
        """
        pass

    def compute_reward(
        self, actions: List[mi.Transition], done: torch.Tensor
    ) -> torch.Tensor:
        """Implementations of this method might vary the reward with the done signal.
        Default gives 0 if the source state is a goal, -1 otherwise. This way the
        goal state should have a value of 0.
        """
        return torch.where(
            done, torch.tensor(0.0, device=done.device), self._minus_one_rewards
        )

    def create_td(
        self, source: TensorDictBase | dict[str, CompatibleType] | None = None
    ) -> TensorDict:
        """Generate an empty tensordict with the correct device and batch-size.."""
        return TensorDict(source or {}, batch_size=self.batch_size, device=self.device)

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

        PlanningEnvironment.assert_1D_batch(tensordict.batch_size)

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

    def get_applicable_transitions(
        self, states: List[mi.State]
    ) -> List[List[mi.Transition]]:
        return [
            self.transitions_for(instance, state)
            for (instance, state) in zip(self._active_instances, states)
        ]

    def _replace_active_instance(self, index: int):
        self._active_instances[index] = next(self._next_active_iterator)

    def _apply_transition_or_stay(
        self,
        i: int,
        transition: Optional[mi.Transition],
        current_states: List[mi.State],
    ):
        if transition:
            return transition.target
        else:
            # Check that there were no transitions available
            state = current_states[i]
            assert (
                len(self.transitions_for(self._active_instances[i], state)) == 0
            ), "Got None transition for state with available transitions."
        return state

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Perform a step in the environment.
        Apply the action, compute a new state, render pixels and determine reward, termination and valid next actions.
        The trajectory is done after a goal state is visited!
        NOTE that EnvBase.step() already checks that the batch_size matches.
        :param td (TensorDict): TensorDict with state and action.
        :returns TensorDict: Output TensorDict.
        """

        current_states: List[mi.State] = tensordict[self._keys.state]

        actions: List[Optional[mi.Transition]] = tensordict[self._keys.action]
        assert isinstance(actions, list)  # batch of chosen-transitions

        # Apply the transition or stay in the current state if none are available.
        next_states = [
            self._apply_transition_or_stay(i, transition, current_states)
            for i, transition in enumerate(actions)
        ]

        # check for termination and reward
        done = torch.tensor(
            [
                self.is_goal(instance, state)
                for (instance, state) in zip(self._active_instances, current_states)
            ],
            dtype=torch.bool,
        )
        reward = self.compute_reward(actions, done)
        assert reward.shape == done.shape

        return self.create_td(
            {
                self._keys.state: as_non_tensor_stack(next_states),
                self._keys.transitions: as_non_tensor_stack(
                    self.get_applicable_transitions(next_states)
                ),
                self._keys.goals: tensordict.get(self._keys.goals),
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
        states: List[mi.State] | NonTensorWrapper | None = None,
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
            to_be_reset_indices: List = partial_reset.nonzero().squeeze(-1).tolist()
        else:
            to_be_reset_indices = list(range(batch_size))

        for i in to_be_reset_indices:
            self._replace_active_instance(i)

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
            }
        )

        return out
