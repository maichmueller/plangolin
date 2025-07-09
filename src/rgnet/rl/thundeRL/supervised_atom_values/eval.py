import csv
import dataclasses
import datetime
import itertools
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch_geometric
import torchrl
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import Logger, WandbLogger
from tensordict import NestedKey, NonTensorStack, TensorDict, TensorDictBase
from tensordict.nn import InteractionType, TensorDictModule, TensorDictModuleBase
from tensordict.tensorclass import NonTensorData
from torch_geometric.data import Batch
from torchrl.data import Composite, NonTensor
from torchrl.envs.utils import set_exploration_type

import xmimir as xmi
from rgnet.encoding import GraphEncoderBase
from rgnet.logging_setup import get_logger
from rgnet.rl.agents import ActorCritic
from rgnet.rl.data_layout import InputData, OutputData
from rgnet.rl.embedding import NonTensorTransformedEnv
from rgnet.rl.embedding.embedding_module import EncodingModule
from rgnet.rl.envs import PlanningEnvironment, SuccessorEnvironment
from rgnet.rl.reward import RewardFunction
from rgnet.rl.thundeRL import AtomValuesCLI
from rgnet.rl.thundeRL.policy_gradient.cli import TestSetup
from rgnet.rl.thundeRL.utils import (
    default_checkpoint_format,
    resolve_checkpoints,
    wandb_id_resolver,
)
from rgnet.utils.misc import as_non_tensor_stack, tolist
from rgnet.utils.plan import Plan

from .lit_module import EmbeddingAndValuator


@dataclasses.dataclass
class ProbabilisticPlanResult(Plan):
    solved: bool
    average_probability: float
    min_probability: float
    rl_return: float
    subgoals: int
    cycles: List[List[xmi.XTransition]]
    # 0 if optimal, positive if higher cost than optimal
    diff_return_to_optimal: Optional[float] = None
    diff_cost_to_optimal: Optional[float] = None

    # cant use dataclasses.asdict(...) because pymimir problems can't be pickled
    def serialize_as_dict(self):
        def transform(k, v):
            if isinstance(v, xmi.XProblem):
                return v.name
            elif k in ["transitions", "cycles"]:
                assert isinstance(v, List) and (
                    len(v) == 0 or (isinstance(v[0], xmi.XTransition))
                )
                return [t.to_string(detailed=True) for t in v]
            return v

        return {
            f.name: transform(f.name, getattr(self, f.name))
            for f in dataclasses.fields(self)
        }


class EncodingTransform(torchrl.envs.Transform):
    """
    Create a transform that merely encodes the states into PyG Data objects.
    """

    enc_transition_key: NestedKey = (
        "encoded_" + PlanningEnvironment.default_keys.transitions
    )
    enc_state_key: NestedKey = "encoded_" + PlanningEnvironment.default_keys.state
    enc_goals_key: NestedKey = "encoded_" + PlanningEnvironment.default_keys.goals

    def __init__(self, env: PlanningEnvironment, encoding_module: EncodingModule):
        super().__init__(
            in_keys=[
                env.keys.state,
                env.keys.goals,
                env.keys.transitions,
            ],
            out_keys=[
                self.enc_state_key,
                self.enc_goals_key,
                self.enc_transition_key,
            ],
        )
        self.env = env
        self.encoding_module = encoding_module

    def _apply_transform(
        self, states_or_transition: NonTensorStack
    ) -> Batch | NonTensorData:
        as_list = tolist(states_or_transition)
        match as_list[0]:
            case list() if isinstance(as_list[0][0], xmi.XLiteral):
                return self.encoding_module(as_list)
            case list() if isinstance(as_list[0][0], xmi.XTransition):
                states, next_states = zip(
                    *(
                        [t.source, t.target]
                        for transitions in as_list
                        for t in transitions
                    )
                )
                batch_states, batch_next_states = (
                    self.encoding_module(states),
                    self.encoding_module(next_states),
                )
                # wrap them in a NonTensorData so TensorDict.set sees
                # a TensorDictBase and skips numpy conversion
                return NonTensorData(
                    data=(batch_states, batch_next_states),
                    batch_size=len(as_list),
                )

            case xmi.XState():
                return self.encoding_module(as_list)
            case _:
                raise TypeError(
                    f"Expected a sequence of XState, XLiteral, or XTransition. Got {type(as_list[0])}"
                )

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """
        Call the transform after an env's `reset` call as well, since we need encodings for initial states too.

        For unknown reasons, the base transform method does nothing on `reset`.
        """
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        """
        Add encoded states spec.
        This is important for _StepMDP validate.
        """
        new_observation_spec = observation_spec.clone()
        for key in [
            self.enc_state_key,
            self.enc_goals_key,
        ]:
            if key not in new_observation_spec:
                new_observation_spec[key] = NonTensor(
                    example_data=torch_geometric.data.Batch(),
                    batched=observation_spec.batch_size[0] > 1,
                )

        new_observation_spec[self.enc_transition_key] = NonTensor(
            example_data=(
                torch_geometric.data.Batch(),
                torch_geometric.data.Batch(),
            ),
            batched=observation_spec.batch_size[0] > 1,
        )
        return new_observation_spec


class ModelMaker(ABC):
    def __init__(
        self, module: torch.nn.Module, checkpoint_path: Path, *, device, **kwargs
    ):
        self.module = module
        self.checkpoint_path = checkpoint_path
        self.device = device

    @abstractmethod
    def agent(self) -> TensorDictModule:
        pass

    @abstractmethod
    def transformed_env(self, base_env: PlanningEnvironment):
        """
        Return a transformed environment that provides the necessary setup to work with the agent.
        This is used to run rollouts on the environment.
        """
        pass


class SortThenCombine(TensorDictModuleBase):
    """
    Simply sorts the atoms by alphanumeric value and then follows the transitions of the first atom after sorting.

    This is used to combine the optimal goal decisions per atom to an overall goal-decision.
    """

    def __init__(
        self,
        in_keys: NestedKey | Sequence[NestedKey],
        out_keys: NestedKey | Sequence[NestedKey],
    ):
        self.in_keys = (in_keys,) if isinstance(in_keys, str) else in_keys
        self.out_keys = (out_keys,) if isinstance(out_keys, str) else out_keys
        super().__init__()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        relevant = tensordict[self.in_keys[0]]
        chosen_atom = relevant.sorted_keys[0]
        return tensordict.set(self.out_keys[0], relevant[chosen_atom])


class RLAtomValuatorActor(TensorDictModule):
    out_key = PlanningEnvironment.default_keys.action
    atom_key = "best_transitions_per_goal"

    def __init__(
        self,
        module: EmbeddingAndValuator,
        in_keys: NestedKey,
        action_key: NestedKey = out_key,
        goal_combiner: torch.nn.Module = SortThenCombine(
            in_keys=atom_key, out_keys="selected"
        ),
    ):
        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=action_key,
        )
        # module to combine the optimal goal decisions per atom to an overall goal-decision
        # (e.g. choose the most frequent best transition, or select one atom first, then another, etc...)
        self.goal_combiner = goal_combiner

    def forward(
        self,
        tensordict: TensorDictBase,
        *args,
        tensordict_out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        # we assume the tensordict has the encoded state
        states, successor_batch, transitions_stack, goals_stack = (
            tensordict.get(k) for k in self.in_keys
        )
        # unpacking `NonTensorStack`s is not fun
        transitions_list: list[list[xmi.XTransition]] = [
            ts.data for ts in transitions_stack.tensordicts
        ]
        goal_list: list[list[xmi.XLiteral]] = [
            gs.data for gs in goals_stack.tensordicts
        ]
        # neither is NonTensorData
        states = states.data
        _, successor_batch = successor_batch.data
        num_successors = list(map(len, transitions_list))
        selected_td = None
        for goals, transitions, num_succ in zip(
            goal_list, transitions_list, num_successors
        ):
            atoms: list[xmi.XAtom] = [goal.atom for goal in goals]
            current_out, output_info = self.module(
                states, atoms=atoms, provide_output_metadata=True
            )
            successor_out, succ_output_info = self.module(
                successor_batch, atoms=atoms, provide_output_metadata=True
            )
            predicates = set(atom.predicate.name for atom in atoms)
            for predicate in predicates:
                current_vals = current_out[predicate].flatten()
                successor_vals = successor_out[predicate].flatten()
                num_atoms = len(current_vals)
                diff = successor_vals - current_vals.repeat(num_succ)
                # Reshape successor values into [num_succ, num_atoms] and select argmax per atom
                successor_vals = successor_vals.view(num_succ, num_atoms)
                successor_vals_masked = torch.where(
                    successor_vals < 0.1, successor_vals, -torch.inf
                )
                best_transition_per_atom = successor_vals_masked.argmax(dim=0)
                # Gather the corresponding max values
                atom_index = torch.arange(num_atoms, device=self.device)
                best_value_per_atom = successor_vals_masked[
                    best_transition_per_atom, atom_index
                ]
                # Compute per-atom improvements and gather
                diffs = diff.view(num_succ, num_atoms)
                best_value_improvement_per_atom = diffs[
                    best_transition_per_atom, atom_index
                ]
                selected_td = TensorDict(batch_size=tensordict.batch_size)
                alternatives_td = TensorDict(batch_size=tensordict.batch_size)
                for i in range(num_atoms):
                    atom = output_info[predicate][i].atom
                    selected_td[atom] = TensorDict(
                        dict(
                            transition_index=best_transition_per_atom[i].view(
                                tensordict.batch_size
                            ),
                            transition=NonTensorData(
                                transitions[best_transition_per_atom[i].item()],
                                batch_size=tensordict.batch_size,
                            ),
                            value=best_value_per_atom[i].view(tensordict.batch_size),
                            current_value=current_vals[i].view(tensordict.batch_size),
                            value_improvement=best_value_improvement_per_atom[i].view(
                                tensordict.batch_size
                            ),
                            # )
                            # )
                            # alternatives_td[atom] = TensorDict(
                            #     dict(
                            all_transitions=transitions,
                            all_values=current_vals[i].view(tensordict.batch_size),
                            all_successor_values=NonTensorData(
                                tolist(successor_vals[:, i]),
                                batch_size=tensordict.batch_size,
                            ),
                        )
                    )
        if selected_td is None:
            get_logger(__name__).warning(
                "No transitions were selected. Check that the env transforms operate as expected. Given TensorDict:\n%s",
                tensordict,
            )
            raise RuntimeError("No transitions were selected.")
        tensordict.set(
            self.atom_key,
            selected_td,
        )
        selected_td = self.goal_combiner(tensordict)
        # set the action key to the selected transition
        tensordict.set(
            self.out_keys[0],
            as_non_tensor_stack([selected_td["selected", "transition"]]),
        )
        return tensordict


class AtomValueModelMaker(ModelMaker):
    """
    Everything produced by a specific model is stored in a ModelResults
    """

    def __init__(
        self,
        atom_value_module: EmbeddingAndValuator,
        checkpoint_path: Path,
        encoder: GraphEncoderBase,
        *,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        # we cant do strict=True, since validation_hooks are often present in the state dict
        atom_value_module.load_state_dict(checkpoint["state_dict"], strict=False)

        self.encoding_module = EncodingModule(
            encoder=encoder,
        ).to(self.device)
        super().__init__(
            module=atom_value_module,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def agent(self) -> TensorDictModule:
        agent = RLAtomValuatorActor(
            self.module,
            in_keys=[
                EncodingTransform.enc_state_key,
                EncodingTransform.enc_transition_key,
                PlanningEnvironment.default_keys.transitions,
                PlanningEnvironment.default_keys.goals,
            ],
        )
        return agent.to(self.device)

    def transformed_env(self, base_env):
        return NonTensorTransformedEnv(
            env=base_env,
            transform=EncodingTransform(
                env=base_env, encoding_module=self.encoding_module
            ),
            cache_specs=True,
            device=self.device,
        )


class MultiCheckpointEvaluation:
    """
    A class that allows to evaluate multiple checkpoints of a model.
    It is used to load the model from a checkpoint and then run it on a given environment.
    """

    def __init__(
        self,
        evaluator,
        module: torch.nn.Module,
        model_maker: type[ModelMaker],
        model_maker_kwargs: Dict[str, Any] | None = None,
        checkpoints: Sequence[Path] | None = None,
        out_data: OutputData | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.evaluator = evaluator
        self._current_model: ModelMaker = None
        self._current_checkpoint: Path = None
        if checkpoints is None:
            assert (
                out_data is not None
            ), "If no checkpoints are provided, out_data must be provided to resolve checkpoints."
            checkpoints, last_checkpoint = resolve_checkpoints(out_data)
        else:
            last_checkpoint = None
            assert isinstance(checkpoints, Sequence)
            if len(checkpoints) == 0:
                warnings.warn("Provided an empty list as checkpoint_paths")
            assert all(
                isinstance(ckpt, Path) and ckpt.is_file() and ckpt.suffix == ".ckpt"
                for ckpt in checkpoints
            )
        self._checkpoints: Sequence[Path] = checkpoints
        self.device = device
        self.module = module.to(self.device)
        self.model_maker = model_maker
        self.model_maker_kwargs = model_maker_kwargs
        self._model_for_checkpoint: Dict[Path, ModelMaker] = dict()
        self.load_checkpoint(
            last_checkpoint if last_checkpoint is not None else checkpoints[-1]
        )

    def load_checkpoint(self, checkpoint_path: Path) -> ModelMaker:
        if checkpoint_path not in self._model_for_checkpoint:
            self._model_for_checkpoint[checkpoint_path] = self.model_maker(
                self.module,
                checkpoint_path,
                device=self.device,
                **(self.model_maker_kwargs or {}),
            )
        return self._model_for_checkpoint[checkpoint_path]

    @property
    def model(self):
        return self._current_model

    @property
    def current_checkpoint(self):
        return self._current_checkpoint

    @current_checkpoint.setter
    def current_checkpoint(self, new_checkpoint: Path):
        self.load_checkpoint(new_checkpoint)

    @property
    def checkpoints(self):
        return self._checkpoints


class ValueSearchEval:
    env_keys = PlanningEnvironment.default_keys

    def __init__(
        self,
        test_setup: TestSetup,
        reward_function: RewardFunction,
        device: torch.device = torch.device("cpu"),
        gamma: float = 1.0,
    ):
        self._current_model = None
        self.test_setup = test_setup
        self.device: torch.device = device
        self.gamma: float = gamma
        self.reward_function: RewardFunction = reward_function
        self._model_for_checkpoint: Dict[Path, ModelMaker] = dict()
        self._successor_env_for_problem: Dict[xmi.XProblem, SuccessorEnvironment] = (
            dict()
        )

    def successor_env_for_problem(self, problem: xmi.XProblem) -> SuccessorEnvironment:
        generator = xmi.XSuccessorGenerator(problem)
        return SuccessorEnvironment(
            generators=[generator],
            reward_function=self.reward_function,
            batch_size=torch.Size((1,)),
        )

    def rollout_on_env(
        self,
        base_env: PlanningEnvironment,
        model: ModelMaker,
        initial_state: xmi.XState | None = None,
        max_steps: int | None = None,
    ):
        env = model.transformed_env(base_env)
        agent = model.agent().to(self.device)

        cycle_transform = CycleAvoidingTransform(self.env_keys.transitions)
        if self.test_setup.avoid_cycles:
            env = NonTensorTransformedEnv(
                env=env,
                transform=cycle_transform,
            )
            env = NonTensorTransformedEnv(
                env=env,
                transform=NoTransitionTruncationTransform(
                    self.env_keys.transitions,
                    self.env_keys.done,
                    self.env_keys.truncated,
                ),
            )
        initial = (
            env.reset(states=[initial_state] * env.batch_size[0])
            if initial_state
            else None
        )
        with set_exploration_type(InteractionType.MODE), torch.no_grad():
            return env.rollout(
                max_steps=max_steps or self.test_setup.max_steps,
                policy=agent,
                tensordict=initial,
            )

    def rollout_on_problem(self, problem: xmi.XProblem, **kwargs):
        base_env = self.successor_env_for_problem(problem)
        return self.rollout_on_env(base_env, **kwargs)

    def analyze(
        self,
        problem: xmi.XProblem,
        rollout: TensorDictBase,
        optimal_plan: Optional[Plan] = None,
    ) -> ProbabilisticPlanResult:
        problem: xmi.XProblem
        # Assert we only have one batch entry and the time dimension is the last
        assert rollout.batch_size[0] == 1
        assert rollout.names[-1] == "time"
        transitions = list(
            itertools.takewhile(
                lambda t: not t.source.is_goal(),
                rollout["action"][0],
            )
        )
        rl_return, cost = compute_return(self.gamma, transitions)
        cycles = analyze_cycles(transitions)
        plan_result = ProbabilisticPlanResult(
            problem=problem,
            solved=rollout[("next", "terminated")].any().item(),
            average_probability=1.0,
            min_probability=1.0,
            transitions=transitions,
            rl_return=round(rl_return, 3),
            cost=cost,
            subgoals=len(transitions),
            cycles=cycles,
        )
        if optimal_plan is not None:
            rl_return_optimal, cost_optimal = compute_return(
                self.gamma, optimal_plan.transitions
            )
            plan_result.diff_cost_to_optimal = plan_result.cost - cost_optimal
            plan_result.diff_return_to_optimal = rl_return - rl_return_optimal
            for i, (plan_step, optimal_plan_step) in enumerate(
                zip(plan_result.transitions, optimal_plan.transitions)
            ):
                if not plan_step.target.semantic_eq(optimal_plan_step.target):
                    print("Deviated from optimal plan at step ", str(i))
                    break
        return plan_result


def compute_return(gamma, transitions):
    rl_return = 0.0
    cost = 0.0
    step = 0
    for transition in transitions:
        if isinstance(transition.action, Sequence):
            rl_return += sum(
                gamma**i * (-action.cost)
                for i, action in enumerate(transition.action, start=step)
            )
            cost += sum(action.cost for action in transition.action)
            step += len(transition.action)
        else:
            rl_return += gamma**step * transition.action.cost
            cost += transition.action.cost
            step += 1
    return rl_return, cost


def analyze_cycles(transitions):
    """
    Analyze the transitions and return the cycles that were made.
    A cycle is a list of transitions that lead to a previously visited state.
    The cycles are grouped by the level of decision-making, e.g., subgoal cycles
    are on the level of subgoals, while primitive action cycles are on the level of primitive actions.
    """
    visited: Set[xmi.XState] = set()
    cycles: List[List[xmi.XTransition]] = []
    current_cycle: List[xmi.XTransition] = []

    for transition in transitions:
        if transition.source in visited:
            # we have a cycle
            new_cycle = [transition]
            if len(current_cycle) > 0:
                cycles.append(current_cycle + new_cycle)
            current_cycle = new_cycle
        else:
            visited.add(transition.source)
            current_cycle.append(transition)
    return cycles


class StochasticPolicy(torch.nn.Module, Callable[[TensorDictBase], TensorDictBase]):
    def __init__(
        self,
        probs_list: List[torch.Tensor],
        problem: xmi.XProblem,
        env_keys: PlanningEnvironment.AcceptedKeys,
        idx_of_state: Callable[[TensorDictBase], List[int]] | xmi.XStateSpace | str,
    ):
        super().__init__()
        self.probs_list = probs_list
        self.problem = problem
        self.env_keys = env_keys
        if isinstance(idx_of_state, str):
            td_key = idx_of_state
            self.idx_of_state = lambda td: td[td_key]
        elif isinstance(idx_of_state, xmi.XStateSpace):
            self.idx_of_state = lambda td: [s.index for s in td[self.env_keys.state]]
        else:
            self.idx_of_state = idx_of_state

    def verify_instance(self, instance):
        if isinstance(instance, Tuple):
            assert isinstance(instance[1], xmi.XProblem)
            return instance[1] == self.problem
        elif isinstance(instance, xmi.XStateSpace):
            return instance.problem == self.problem
        raise RuntimeError(
            f"Got instance {instance} which was neither of type "
            f"Tuple[SuccessorGenerator,Problem] nor StateSpace"
        )

    def forward(self, tensordict: TensorDict):
        assert all(self.verify_instance(i) for i in tensordict[self.env_keys.instance])
        indices: List[int] = self.idx_of_state(tensordict)
        batched_probs: List[torch.Tensor] = [self.probs_list[idx] for idx in indices]
        actions = []
        log_probs = []
        selected_probs = []
        for probs in batched_probs:
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = dist.sample()
            log_probs.append(dist.log_prob(action_idx))
            selected_probs.append(probs[action_idx])
            actions.append(action_idx)
        actions = [
            ts[idx] for ts in tensordict[self.env_keys.transitions] for idx in actions
        ]
        tensordict[self.env_keys.action] = as_non_tensor_stack(actions)
        tensordict[ActorCritic.default_keys.probs] = as_non_tensor_stack(selected_probs)
        tensordict[ActorCritic.default_keys.log_probs] = torch.stack(log_probs)

        return tensordict


class CycleAvoidingTransform(torchrl.envs.Transform):
    """
    Keep track of current states and filter out any transition that leads to a visited state.
    Each batch entry is treated independently.
    """

    def __init__(self, transitions_key: NestedKey):
        super().__init__(in_keys=[transitions_key], out_keys=[transitions_key])
        self.transition_key = transitions_key
        self.visited: Dict[int, Set[xmi.XState]] = defaultdict(set)

    def _apply_transform(
        self, batched_transitions: List[List[xmi.XTransition]] | NonTensorStack
    ) -> NonTensorStack:
        batched_transitions = tolist(batched_transitions)
        filtered_transitions: list[list[xmi.XTransition]] = []
        for batch_idx, transitions in enumerate(batched_transitions):
            assert (
                len(transitions) > 0
            ), "Environment returned state without outgoing transitions."
            # all transitions should have the same source state
            self.visited[batch_idx].add(transitions[0].source)
            filtered: List[xmi.XTransition] = [
                t for t in transitions if t.target not in self.visited[batch_idx]
            ]
            filtered_transitions.append(filtered)

        return as_non_tensor_stack(filtered_transitions)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self.visited.clear()
        if tensordict.get("_reset", default=None) is not None:
            raise RuntimeError("Transform is not implemented with partial resets")
        return self._call(tensordict_reset)


class NoTransitionTruncationTransform(torchrl.envs.Transform):
    """
    After CycleAvoidingTransform has filtered out all successors,
    force a truncation (i.e. set `truncated=True` and thus `done=True`)
    on any batch index where `transitions` is empty.
    """

    def __init__(
        self,
        transitions_key: NestedKey,
        done_key: NestedKey,
        truncated_key: NestedKey,
    ):
        # We do NOT pass anything to in_keys / out_keys, because we override _step directly.
        super().__init__(
            in_keys=None, out_keys=None, in_keys_inv=None, out_keys_inv=None
        )
        self.transitions_key = transitions_key
        self.done_key = done_key
        self.truncated_key = truncated_key

    def _step(
        self,
        tensordict: TensorDict,  # the TensorDict at time t
        next_tensordict: TensorDict,  # the TensorDict at time t+1 (after env.step)
    ) -> TensorDict:
        """
        Called after the base env (and any previous transforms) have stepped.
        We look at `next_tensordict[self.transitions_key]` and, wherever that list is empty,
        we force `truncated=True` and then `done=True`.
        """
        batched_transitions = next_tensordict.get(self.transitions_key)
        transitions_list = tolist(batched_transitions)  # now a Python list of lists

        batch_size = len(transitions_list)
        device = next_tensordict.device

        truncated = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        for i in range(batch_size):
            if len(transitions_list[i]) == 0:
                truncated[i, 0] = True

        # read existing `done` (if absent, assume all False)
        if next_tensordict.get(self.done_key, None) is None:
            done_existing = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=device
            )
        else:
            done_existing = next_tensordict.get(self.done_key)

        new_done = done_existing | truncated
        next_tensordict.set(self.truncated_key, truncated)
        next_tensordict.set(self.done_key, new_done)

        return next_tensordict


class EvalAtomValuesCLI(AtomValuesCLI):
    def add_arguments_to_parser_impl(self, parser: LightningArgumentParser) -> None:
        # fit subcommand adds this value to the config
        parser.add_argument("--ckpts", type=Optional[Path | list[Path]], default=None)
        parser.add_argument("--device", type=str, default="cpu")
        parser.link_arguments(
            "data_layout.output_data",
            "trainer.logger.init_args.id",
            compute_fn=wandb_id_resolver,
            apply_on="instantiate",
        )


def eval_model(
    cli: AtomValuesCLI,
    module: EmbeddingAndValuator,
    logger: Logger,
    input_data: InputData,
    output_data: OutputData,
    test_setup: TestSetup,
):
    """
    Run the learned agent of every test problem specified in the input_data.
    The agent is run once on each problem until either max_steps are reached or a terminal state is encountered.
    There is currently no GPU support as the memory transfer is often more significant as
    the relatively small network forward passes.
    The action can either be sampled from the probability distribution using ExplorationType.RANDOM
    or the argmax is used (ExplorationType.MODE).
    There is no cycle detection implemented.
    Note that the terminated signal is only emitted after a transition from a goal state.
    The output is saved as csv under the out directory of OutputData as results_{epoch}_{step}.csv
    referencing the epoch and step form the loaded checkpoint.

    :param cli: The CLI instance used to run the agent.
    :param module: An agent instance. The weights for the agent will be loaded from a checkpoint.
    :param logger: If a WandbLogger is passed, the results are uploaded as table.
    :param input_data: InputData which should specify at least one test_problem.
    :param output_data: OutputData pointing to the checkpoint containing the learned weights for the agent.
    :param test_setup: Extra parameter for testing the agent.
    """
    if not input_data.test_problems:
        raise ValueError("No test instances provided")
    estimator_config = cli.config_init.get("estimator_config")
    if estimator_config is None:
        gamma = 1.0
    else:
        gamma = estimator_config.gamma
    value_search = ValueSearchEval(
        test_setup,
        reward_function=cli.config_init["reward"],
        gamma=gamma,
        device=torch.device(cli.config_init["device"]),
    )
    multi_checkpoint_routine = MultiCheckpointEvaluation(
        value_search,
        module,
        AtomValueModelMaker,
        out_data=output_data,
        model_maker_kwargs={"encoder": cli.config_init.encoder},
        device=torch.device(cli.config_init["device"]),
    )
    for checkpoint_path in multi_checkpoint_routine.checkpoints:
        model = multi_checkpoint_routine.load_checkpoint(checkpoint_path)
        epoch, step = default_checkpoint_format(checkpoint_path.name)
        logger = get_logger(__name__)
        logger.info(f"Using checkpoint with {epoch=}, {step=}")

        test_results: List[ProbabilisticPlanResult] = []
        test_instances = input_data.test_problems
        counter = itertools.count(0)
        for test_problem in test_instances:
            logger.info(
                f"Running rollout (max steps {test_setup.max_steps}) for problem {test_problem.name, test_problem.filepath}."
            )
            start = time.time()
            rollout = value_search.rollout_on_problem(test_problem, model=model)
            logger.info(
                f"Rollout completed in {datetime.timedelta(seconds=int(time.time() - start))}"
            )
            analyzed_data: ProbabilisticPlanResult = value_search.analyze(
                test_problem,
                rollout,
                optimal_plan=input_data.plan_by_problem.get(test_problem),
            )
            test_results.append(analyzed_data)
            plan_string = "\n".join(
                f"{t.action.str(for_plan=True)}" for t in analyzed_data.transitions
            )
            logger.info(
                f"Problem {test_problem.name}\n"
                f"solved: {analyzed_data.solved}\n"
                f"return: {analyzed_data.rl_return}\n"
                f"cost: {analyzed_data.cost}\n"
                f"subgoals: {analyzed_data.subgoals}\n"
                f"cycles: {len(analyzed_data.cycles)}\n"
                f"plan: \n{plan_string}"
            )
            logger.info(
                f"Problems remaining: {len(test_instances) - next(counter)} / {len(test_instances)}"
            )
        solved = sum(p.solved for p in test_results)
        logger.info(f"Solved {solved} out of {len(test_results)}")

        results_name = f"results_epoch={epoch}-step={step}"
        results_file = output_data.out_dir / (results_name + ".csv")
        plan_results_as_dict = [
            plan_result.serialize_as_dict() for plan_result in test_results
        ]
        with open(results_file, "w") as f:
            writer = csv.DictWriter(
                f,
                plan_results_as_dict[0].keys(),
            )
            writer.writeheader()
            writer.writerows(plan_results_as_dict)
        logger.info("Saved results to " + str(results_file))

        if isinstance(logger, WandbLogger) and logger.experiment is not None:
            table_data = [
                list(plan_dict.values())  # dicts retain insertion order after 3.7
                for plan_dict in plan_results_as_dict
            ]
            logger.log_table(
                key=results_name,
                columns=list(plan_results_as_dict[0].keys()),
                data=table_data,
                step=step,
            )
            logger.log_metrics({"solved": solved}, step=step)
            logger.finalize(status="success")


def eval_lightning_agent_cli():
    # overwrite this because it might be set in the config.yaml.
    sys.argv.extend(["--data_layout.output_data.ensure_new_out_dir", "false"])
    # needs to be set to avoid loading all the training and validation data
    sys.argv.extend(["--data.skip", "true"])
    # Should be set to avoid overwriting the previous run with the same id (workaround because we can't set the default)
    sys.argv.extend(["--trainer.logger.init_args.resume", "true"])
    cli = EvalAtomValuesCLI(run=False)
    module: EmbeddingAndValuator = cli.model.embedder_and_valuator
    in_data: InputData = cli.datamodule.data
    out_data = cli.config_init["data_layout.output_data"]
    test_setup: TestSetup = cli.config_init["test_setup"]
    eval_model(
        cli=cli,
        module=module,
        logger=cli.trainer.logger,
        input_data=in_data,
        output_data=out_data,
        test_setup=test_setup,
    )


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")
    eval_lightning_agent_cli()
