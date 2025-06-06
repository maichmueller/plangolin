import copy
import csv
import dataclasses
import functools
import itertools
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch_geometric as pyg
import torchrl
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import Logger, WandbLogger
from tensordict import NestedKey, NonTensorStack, TensorDict, TensorDictBase
from tensordict.nn import InteractionType, TensorDictModule
from torch import Tensor
from torchrl.envs.utils import set_exploration_type

import xmimir as xmi
from rgnet.encoding import HeteroGraphEncoder
from rgnet.logging_setup import tqdm
from rgnet.models import PyGHeteroModule
from rgnet.rl.agents import ActorCritic
from rgnet.rl.data_layout import InputData, OutputData
from rgnet.rl.embedding import (
    EmbeddingModule,
    EmbeddingTransform,
    NonTensorTransformedEnv,
)
from rgnet.rl.envs import (
    ExpandedStateSpaceEnv,
    PlanningEnvironment,
    SuccessorEnvironment,
)
from rgnet.rl.reward import RewardFunction
from rgnet.rl.thundeRL import PolicyGradientCLI
from rgnet.rl.thundeRL.policy_gradient.cli import TestSetup
from rgnet.rl.thundeRL.policy_gradient.lit_module import PolicyGradientLitModule
from rgnet.rl.thundeRL.utils import (
    default_checkpoint_format,
    resolve_checkpoints,
    wandb_id_resolver,
)
from rgnet.utils.misc import as_non_tensor_stack, tolist
from rgnet.utils.object_embeddings import ObjectEmbedding
from rgnet.utils.plan import Plan
from xmimir import iw


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


class ModelData:
    """
    Everything produced by a specific model is stored in a ModelResults
    """

    def __init__(
        self,
        policy_gradient_lit_module: PolicyGradientLitModule,
        checkpoint_path: Path,
        experiment_analyzer,
    ):
        self._parent: RLPolicySearchEvaluator = experiment_analyzer

        policy_gradient_module = copy.deepcopy(policy_gradient_lit_module)
        checkpoint = torch.load(
            checkpoint_path, map_location=self._parent.device, weights_only=False
        )
        # we cant do strict=True, since validation_hooks are often present in the state dict
        policy_gradient_module.load_state_dict(checkpoint["state_dict"], strict=False)
        embedding = EmbeddingModule(
            encoder=HeteroGraphEncoder(self._parent.in_data.domain),
            gnn=policy_gradient_module.gnn,
            device=self._parent.device,
        )
        self.agent: ActorCritic = policy_gradient_module.actor_critic
        # TODO fix embedding from agent, can't mix properties and torch.nn.Module
        self.agent._embedding_module = embedding
        self.policy_gradient_module: PolicyGradientLitModule = policy_gradient_module
        self.gnn: PyGHeteroModule = self.policy_gradient_module.gnn
        self.policy: TensorDictModule = self.agent.as_td_module(
            self._parent.env_keys.state,
            self._parent.env_keys.transitions,
            self._parent.env_keys.action,
            add_probs=True,
            # out_successor_embeddings=True,
        )

        # All states over a single space have the same number of objects.
        # We can simply save the dense embeddings without the mask.
        # Shape [space.num_states(), num_objects, hidden_size]
        self._embedding_for_space: Dict[xmi.XStateSpace, torch.Tensor] = dict()

        # Computed probs for space
        self._computed_probs_list_for_space: Dict[
            xmi.XStateSpace, List[torch.Tensor]
        ] = dict()
        self._action_indices_for_space: Dict[xmi.XStateSpace, torch.Tensor] = dict()
        self._log_probs_for_space: Dict[xmi.XStateSpace, torch.Tensor] = dict()

        # Computed values for space
        self._computed_values_for_space: Dict[xmi.XStateSpace, torch.Tensor] = dict()

    def _compute_probs_for_space(self, space: xmi.XStateSpace):
        with set_exploration_type(InteractionType.MODE):
            embeddings: Tensor = self.embedding_for_space(space)
            successor_indices: dict[xmi.XState, list[int]] = (
                self._parent.successor_indices(space)
            )
            successor_indices_list: List[torch.Tensor] = [
                torch.tensor(successor_indices[s]) for s in space
            ]
            num_successors: torch.Tensor = torch.tensor(
                [len(ls) for ls in successor_indices_list],
                dtype=torch.long,
                device=self._parent.device,
            )
            successor_embeddings = torch.cat(
                [embeddings[indices] for indices in successor_indices_list], dim=0
            )
            assert num_successors.size(0) == embeddings.size(0)
            assert successor_embeddings.size(0) == num_successors.sum()

            batched_probs, action_indices, log_probs = self.agent.embedded_forward(
                ObjectEmbedding(
                    embeddings,
                    torch.ones(
                        size=embeddings.shape[:-1],
                        dtype=torch.bool,
                        device=self._parent.device,
                    ),
                ),
                ObjectEmbedding(
                    successor_embeddings,
                    torch.ones(
                        size=successor_embeddings.shape[:-1],
                        dtype=torch.bool,
                        device=self._parent.device,
                    ),
                ),
                num_successors,
            )
            self._computed_probs_list_for_space[space] = batched_probs
            self._action_indices_for_space[space] = action_indices
            self._log_probs_for_space[space] = log_probs

    def computed_probs_list_for_space(self, space: xmi.XStateSpace):
        if space not in self._computed_probs_list_for_space:
            self._compute_probs_for_space(space)
        return self._computed_probs_list_for_space[space]

    def action_indices_for_space(self, space: xmi.XStateSpace):
        if space not in self._action_indices_for_space:
            self._compute_probs_for_space(space)
        return self._action_indices_for_space[space]

    def log_probs_for_space(self, space: xmi.XStateSpace):
        if space not in self._log_probs_for_space:
            self._compute_probs_for_space(space)
        return self._log_probs_for_space[space]

    def computed_values_for_space(self, space: xmi.XStateSpace):
        if space not in self._computed_values_for_space:
            embeddings = self.embedding_for_space(space)
            values = self.agent.value_operator.module(embeddings)
            self._computed_values_for_space[space] = values
        return self._computed_values_for_space[space]

    def embedding_for_space(self, space: xmi.XStateSpace):
        if space not in self._embedding_for_space:
            logging.info(
                f"Computing embedding for whole state space {space} with {len(space)} states."
                f" This might take a while."
            )
            self._embedding_for_space[space] = self.agent.embedding_module(
                list(space)
            ).dense_embedding
        return self._embedding_for_space[space]

    def transformed_embedding_env(self, base_env):
        return NonTensorTransformedEnv(
            env=base_env,
            transform=EmbeddingTransform(
                current_embedding_key=self.agent.keys.current_embedding,
                env=base_env,
                embedding_module=self.agent.embedding_module,
            ),
            cache_specs=True,
            device=self._parent.device,
        )


class RLPolicySearchEvaluator:
    env_keys = PlanningEnvironment.default_keys

    def __init__(
        self,
        lightning_agent: PolicyGradientLitModule,
        in_data: InputData,
        out_data: OutputData,
        test_setup: TestSetup,
        reward_function: RewardFunction,
        checkpoints_paths: List[Path] | None = None,
        device: torch.device = torch.device("cpu"),
        gamma: float = 1.0,
    ):
        self._current_model = None
        self.in_data = in_data
        self.out_data = out_data
        self.test_setup = test_setup
        self.agent_keys = lightning_agent.actor_critic.keys
        self.device: torch.device = device
        self.gamma: float = gamma
        self.reward_function: RewardFunction = reward_function
        self._model_for_checkpoint: Dict[Path, ModelData] = dict()
        self._policy_gradient_lit_module = lightning_agent.to(self.device)
        if checkpoints_paths is None:
            checkpoints_paths, last_checkpoint = resolve_checkpoints(out_data)
        else:
            last_checkpoint = None
            assert isinstance(checkpoints_paths, List)
            if len(checkpoints_paths) == 0:
                warnings.warn("Provided an empty list as checkpoint_paths")
            assert all(
                isinstance(ckpt, Path) and ckpt.is_file() and ckpt.suffix == ".ckpt"
                for ckpt in checkpoints_paths
            )

        self._current_model: ModelData
        self._current_checkpoint: Path
        self.load_checkpoint(
            last_checkpoint if last_checkpoint is not None else checkpoints_paths[-1]
        )
        self._checkpoints: List[Path] = checkpoints_paths

        self._successor_env_for_problem: Dict[xmi.XProblem, SuccessorEnvironment] = (
            dict()
        )
        self._expanded_env_for_problem: Dict[xmi.XProblem, ExpandedStateSpaceEnv] = (
            dict()
        )
        # Paths will be resolved on demand, but the list itself is consistent.
        # Can be None if no probs were saved (:class: `ProbsStoreCallback` not used).
        # The lists are sorted by epoch.
        self._stored_probs_for_problem: (
            Dict[
                xmi.XProblem,
                List[Tuple[int, Path]] | List[Tuple[int, List[torch.Tensor]]],
            ]
            | None
        ) = self._load_stored_probs()
        self._successor_indices: Dict[xmi.XStateSpace, Dict[xmi.XState, List[int]]] = (
            dict()
        )
        self._expanded_envs: Dict[xmi.XStateSpace, pyg.data.Data] = dict()

    def load_checkpoint(self, checkpoint_path: Path):
        if checkpoint_path not in self._model_for_checkpoint:
            model = ModelData(self._policy_gradient_lit_module, checkpoint_path, self)
            self._model_for_checkpoint[checkpoint_path] = model
        self._current_checkpoint = checkpoint_path
        self._current_model = self._model_for_checkpoint[checkpoint_path]
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

    @property
    def model(self):
        return self._current_model

    # Shortcut for simplicity
    def space(self, problem: xmi.XProblem):
        return self.in_data.get_or_load_space(problem)

    def successor_indices(self, space: xmi.XStateSpace):
        if space not in self._successor_indices:
            self._successor_indices[space] = {
                s: [t.target.index for t in space.forward_transitions(s)] for s in space
            }
        return self._successor_indices[space]

    @functools.cache
    def successor_env_for_problem(self, problem: xmi.XProblem) -> SuccessorEnvironment:
        if problem not in self._successor_env_for_problem:
            if self.test_setup.iw_search is not None:
                generator = iw.IWSuccessorGenerator(self.test_setup.iw_search, problem)
            else:
                generator = xmi.XSuccessorGenerator(problem)
            base_env = SuccessorEnvironment(
                generators=[generator],
                reward_function=self.reward_function,
                batch_size=torch.Size((1,)),
            )
            self._successor_env_for_problem[problem] = base_env
        return self._successor_env_for_problem[problem]

    def expanded_env_for_problem(self, problem: xmi.XProblem) -> ExpandedStateSpaceEnv:
        space = self.in_data.get_or_load_space(problem)
        if space is None:
            raise ValueError(
                f"Could not find space for problem {problem}."
                f"Try increasing max_expanded"
            )
        if problem not in self._expanded_env_for_problem:
            base_env = ExpandedStateSpaceEnv(
                space=space,
                reward_function=self.reward_function,
                batch_size=torch.Size((1,)),
            )
            self._expanded_env_for_problem[problem] = base_env
        return self._expanded_env_for_problem[problem]

    def rollout_on_env(
        self,
        base_env: PlanningEnvironment,
        initial_state: xmi.XState | None = None,
        exploration_type: InteractionType | None = None,
        max_steps: int | None = None,
    ):
        env = self.model.transformed_embedding_env(base_env)
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
        exploration_type = exploration_type or self.test_setup.exploration_type
        with set_exploration_type(exploration_type), torch.no_grad():
            return env.rollout(
                max_steps=max_steps or self.test_setup.max_steps,
                policy=self.model.policy,
                tensordict=initial,
            )

    def rollout_on_problem(self, problem: xmi.XProblem, **kwargs):
        base_env = self.successor_env_for_problem(problem)
        return self.rollout_on_env(base_env, **kwargs)

    def _load_stored_probs(
        self,
    ) -> Optional[Dict[xmi.XProblem, List[Tuple[int, Path]]]]:
        probs_store_callback_default_name = "actor_probs"

        def problem_matching_name(name: str, problems: Iterable[xmi.XProblem] | None):
            return (
                next(
                    (p for p in problems if name in p.name),
                    None,
                )
                if problems is not None
                else None
            )

        def find_problem_for_name(name: str):
            problem_sources = [
                self.in_data.validation_problems,
                self.in_data.problems,
                self.in_data.test_problems,
            ]
            try:
                return next(
                    filter(
                        lambda x: x is not None,
                        (
                            problem_matching_name(name, source)
                            for source in problem_sources
                        ),
                    )
                )
            except StopIteration:
                raise RuntimeError(
                    f"Could not find any problem (train/val/test) that matches the name {name}"
                )

        probs_dir = self.out_data.out_dir / probs_store_callback_default_name
        if not probs_dir.is_dir():
            logging.info(
                f"Could not find actor_probs for experiment at {self.out_data.out_dir}"
            )
            return None
        probs_paths = list(probs_dir.iterdir())
        by_problem: Dict[xmi.XProblem, List[Tuple[int, Path]]] = defaultdict(list)
        for path in probs_paths:
            stem = path.stem.removeprefix(f"{probs_store_callback_default_name}_")
            epoch: str = stem.split("_")[-1]
            dataloader_name = stem.removesuffix(f"_{epoch}")
            problem: xmi.XProblem = find_problem_for_name(dataloader_name)
            by_problem[problem].append((int(epoch), path))
        by_problem = {p: sorted(paths) for (p, paths) in by_problem.items()}
        return by_problem

    def stored_probs_for_problem(
        self, problem: xmi.XProblem
    ) -> List[Tuple[int, List[torch.Tensor]]]:
        if (
            self._stored_probs_for_problem is None
            or problem not in self._stored_probs_for_problem
        ):
            raise ValueError(
                f"Either no probabilities were saved for the problem or not for this problem {problem}"
            )
        epoch_list = self._stored_probs_for_problem[problem]
        if len(epoch_list) == 0:  # e.g., no validation epoch finished
            return []
        if isinstance(epoch_list[0][1], Path):
            loaded_probs_list: List[Tuple[int, List[torch.Tensor]]] = [
                (epoch, torch.load(pfile, map_location=self.device))
                for epoch, pfile in epoch_list
            ]
            self._stored_probs_for_problem[problem] = loaded_probs_list
        return self._stored_probs_for_problem[problem]

    def map_to_probabilistic_plan(
        self, problem: xmi.XProblem, rollout: TensorDictBase
    ) -> ProbabilisticPlanResult:
        problem: xmi.XProblem
        # Assert we only have one batch entry and the time dimension is the last
        assert rollout.batch_size[0] == 1
        assert rollout.names[-1] == "time"
        action_probs = rollout["log_probs"].detach().exp()
        transitions = list(
            itertools.takewhile(
                lambda t: not t.source.is_goal(),
                rollout["action"][0],
            )
        )
        plan_length = len(transitions)
        action_probs = action_probs[:plan_length]
        rl_return, cost = self._compute_return(transitions)
        cycles = self._analyze_cycles(transitions)
        plan_result = ProbabilisticPlanResult(
            problem=problem,
            solved=rollout[("next", "terminated")].any().item(),
            average_probability=round(action_probs.mean().item(), 4),
            min_probability=round(action_probs.min().item(), 4),
            transitions=transitions,
            rl_return=round(rl_return, 3),
            cost=cost,
            subgoals=len(transitions),
            cycles=cycles,
        )
        optimal_plan: Optional[Plan] = self.in_data.plan_by_problem.get(problem)
        if optimal_plan is not None:
            rl_return_optimal, cost_optimal = self._compute_return(
                optimal_plan.transitions
            )
            assert cost_optimal == plan_result.cost
            plan_result.diff_cost_to_optimal = plan_result.cost - cost_optimal
            plan_result.diff_return_to_optimal = rl_return - rl_return_optimal
            for i, (plan_step, optimal_plan_step) in enumerate(
                zip(plan_result.transitions, optimal_plan.transitions)
            ):
                if not plan_step.target.semantic_eq(optimal_plan_step.target):
                    print("Deviated from optimal plan at step ", str(i))
                    break
        return plan_result

    def _compute_return(self, transitions):
        rl_return = 0.0
        cost = 0.0
        step = 0
        for transition in transitions:
            if isinstance(transition.action, Sequence):
                rl_return += sum(
                    self.gamma**i * (-action.cost)
                    for i, action in enumerate(transition.action, start=step)
                )
                cost += sum(action.cost for action in transition.action)
                step += len(transition.action)
            else:
                rl_return += self.gamma**step * transition.action.cost
                cost += transition.action.cost
                step += 1
        return rl_return, cost

    def use_stored_probs_as_policy(
        self, problem: xmi.XProblem, epoch: int | None = None
    ):
        epoch_list = self.stored_probs_for_problem(problem)
        if len(epoch_list) == 0:
            raise ValueError(f"No probabilities were saved for problem {problem}.")
        probs_list: List[torch.Tensor] = (
            epoch_list[-1][1]  # get latest and take list of tensors
            if epoch is None
            else next(probs for (e, probs) in epoch_list if e == epoch)
        )
        if (space := self.in_data.get_or_load_space(problem)) is not None:
            idx_of_state = space
        else:
            idx_of_state = "idx_in_space"
        return StochasticPolicy(
            probs_list, problem, env_keys=self.env_keys, idx_of_state=idx_of_state
        )

    def _analyze_cycles(self, transitions):
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
        filtered_transitions = []
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


class EvalPolicyGradientCLI(PolicyGradientCLI):
    def add_arguments_to_parser_impl(self, parser: LightningArgumentParser) -> None:
        # fit subcommand adds this value to the config
        parser.add_argument("--ckpt_path", type=Optional[Path], default=None)
        parser.add_argument("--device", type=str, default="cpu")
        parser.link_arguments(
            "data_layout.output_data",
            "trainer.logger.init_args.id",
            compute_fn=wandb_id_resolver,
            apply_on="instantiate",
        )


def eval_model(
    cli: PolicyGradientCLI,
    policy_gradient_lit_module: PolicyGradientLitModule,
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
    :param policy_gradient_lit_module: An agent instance. The weights for the agent will be loaded from a checkpoint.
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
    analyzer = RLPolicySearchEvaluator(
        policy_gradient_lit_module,
        in_data=input_data,
        out_data=output_data,
        test_setup=test_setup,
        reward_function=cli.config_init["reward"],
        gamma=gamma,
        device=torch.device(cli.config_init["device"]),
    )
    for checkpoint_path in analyzer.checkpoints:
        analyzer.current_checkpoint = checkpoint_path
        epoch, step = default_checkpoint_format(checkpoint_path.name)
        logging.info(f"Using checkpoint with {epoch=}, {step=}")

        test_instances = input_data.test_problems

        test_results = [
            analyzer.rollout_on_problem(test_problem)
            for test_problem in tqdm(test_instances)
        ]

        analyzed_data: List[ProbabilisticPlanResult] = [
            analyzer.map_to_probabilistic_plan(problem, rollout)
            for problem, rollout in zip(test_instances, test_results)
        ]
        solved = sum(p.solved for p in analyzed_data)
        logging.info(f"Solved {solved} out of {len(analyzed_data)}")

        results_name = f"results_epoch={epoch}-step={step}"
        results_file = output_data.out_dir / (results_name + ".csv")
        plan_results_as_dict = [
            plan_result.serialize_as_dict() for plan_result in analyzed_data
        ]
        with open(results_file, "w") as f:
            writer = csv.DictWriter(
                f,
                plan_results_as_dict[0].keys(),
            )
            writer.writeheader()
            writer.writerows(plan_results_as_dict)
        logging.info("Saved results to " + str(results_file))

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
    cli = EvalPolicyGradientCLI(run=False)
    policy_gradient_lit_module: PolicyGradientLitModule = cli.model
    in_data: InputData = cli.datamodule.data
    out_data = cli.config_init["data_layout.output_data"]
    test_setup: TestSetup = cli.config_init["test_setup"]
    eval_model(
        cli=cli,
        policy_gradient_lit_module=policy_gradient_lit_module,
        logger=cli.trainer.logger,
        input_data=in_data,
        output_data=out_data,
        test_setup=test_setup,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # torch.multiprocessing.set_sharing_strategy("file_system")
    eval_lightning_agent_cli()
