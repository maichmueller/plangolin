import contextlib
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from itertools import chain
from multiprocessing import RawValue
from multiprocessing import synchronize as synchronize
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from tqdm.asyncio import tqdm_asyncio

import xmimir as xmi
from rgnet.algorithms.policy_evaluation_mp import OptimalAtomValuesMP
from rgnet.encoding import GraphEncoderBase
from rgnet.logging_setup import tqdm
from rgnet.rl.envs import (
    ExpandedStateSpaceEnv,
    PlanningEnvironment,
    SuccessorEnvironment,
)
from rgnet.utils.misc import KeyAwareDefaultDict, env_aware_cpu_count, return_true
from xmimir import (
    XAtom,
    XCategory,
    XState,
    XStateSpace,
    XSuccessorGenerator,
    gather_objects,
    parse,
)
from xmimir.iw import CollectorHook, IWSearch
from xmimir.wrappers import CustomProblem, XLiteral, XProblem

from .drive import BaseDrive


def make_atom_ids(problem: XProblem) -> tuple[dict[str, int], list[XAtom]]:
    nr_fluent_atoms = problem.atom_count(XCategory.fluent)
    factor, remainder = divmod(nr_fluent_atoms, 10)
    if remainder > 0:
        factor += 1
    atom_lookup = {}
    all_atoms = []
    for atom in problem.all_atoms(XCategory.fluent):
        atom_lookup[str(atom)] = atom.base.get_index()
        all_atoms.append(atom)
    for atom in problem.all_atoms(XCategory.derived):
        atom_lookup[str(atom)] = 10**factor + atom.base.get_index()
        all_atoms.append(atom)
    return atom_lookup, all_atoms


def atom_value_tensor_to_dict(
    atom_tensor: torch.Tensor,
    index_to_atom_map: Mapping[int, str],
) -> list[dict[str, float]]:
    atom_ids = torch.tensor(list(index_to_atom_map.keys()), dtype=torch.int)
    assert (
        atom_ids == torch.arange(atom_ids.min().item(), atom_ids.max().item() + 1)
    ).all()
    atom_values = []
    if atom_tensor.ndim == 1:
        atom_tensor = atom_tensor.unsqueeze(0)
    for state_atom_values in atom_tensor:
        atom_value_dict = {}
        for j, value in enumerate(state_atom_values):
            atom_value_dict[index_to_atom_map[j]] = value.item()
        atom_values.append(atom_value_dict)
    return atom_values


def atom_value_dict_to_tensor(
    atom_values: Mapping[XState | int, dict[str | XAtom, float]],
    atom_to_index_map: Mapping[str, int],
) -> torch.Tensor:
    atom_ids = torch.tensor(list(atom_to_index_map.values()), dtype=torch.int)
    assert (
        atom_ids == torch.arange(atom_ids.min().item(), atom_ids.max().item() + 1)
    ).all()
    assert torch.unique(atom_ids).numel() == atom_ids.numel(), (
        "Atom IDs must be unique, but found duplicates."
        f" Unique IDs: {torch.unique(atom_ids)}, All IDs: {atom_ids}"
    )
    atom_tensor = torch.full(
        (len(atom_values), len(atom_ids)), torch.inf, dtype=torch.float
    )
    for i, value_dict in enumerate(atom_values.values()):
        atom_indices = [atom_to_index_map[str(atom)] for atom in value_dict.keys()]
        atom_tensor[i, atom_indices] = torch.tensor(
            list(value_dict.values()), dtype=torch.float
        )
    atom_values = atom_tensor
    return atom_values


class AtomValueMethod(Enum):
    DISTANCE_BASED = "distance_based"
    MESSAGE_PASSING = "message_passing"
    IW = "iw"


class AtomDrive(BaseDrive):
    """
    AtomDrive stores for each state s their optimal distance to any atom p, i.e., a value function V(p | s).
    """

    def __init__(
        self,
        *args,
        atom_value_method: AtomValueMethod = AtomValueMethod.MESSAGE_PASSING,
        store_values_as_tensor: bool = True,
        **kwargs,
    ):
        self._atom_to_index_map: dict[str, int] | None = None
        self._index_to_atom_map: dict[int, str] | None = None
        self.atom_value_method = AtomValueMethod(atom_value_method)
        self.store_values_as_tensor = store_values_as_tensor
        super().__init__(*args, **kwargs)

    @property
    def atom_to_index_map(self) -> dict[str, int]:
        if self._atom_to_index_map is None:
            self._atom_to_index_map = self.try_get_data("aux.atom_to_index_map")
        return self._atom_to_index_map

    @property
    def index_to_atom_map(self) -> dict[int, str]:
        if self._index_to_atom_map is None:
            self._index_to_atom_map = {v: k for k, v in self.atom_to_index_map.items()}
        return self._index_to_atom_map

    def _build(self) -> List[HeteroData]:
        env = self.env
        space: xmi.XStateSpace = env.active_instances[0]
        self.problem = space.problem
        encoder = self.encoder_factory(space.problem.domain)
        nr_states: int = len(space)
        logger = self._get_logger()
        logger.info(
            f"Building {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        atom_id_map, atoms = make_atom_ids(space.problem)
        self._atom_to_index_map = atom_id_map
        self._save_aux_data(atom_to_index_map=atom_id_map)
        # Each data object represents one state
        states = list(space)
        atom_values = self._compute_atom_dists(env, states)
        match atom_values:
            case dict():
                atom_values_dict = list(atom_values.values())
                atom_values_tensor = atom_value_dict_to_tensor(
                    atom_values, self.atom_to_index_map
                )
            case torch.Tensor():
                atom_values_dict = atom_value_tensor_to_dict(
                    atom_values, self.atom_to_index_map, self.index_to_atom_map
                )
                atom_values_tensor = atom_values
            case _:
                raise ValueError(f"Unsupported atom values type: {type(atom_values)}")

        batched_data = self._encode(
            env, encoder, atom_values_dict, atom_values_tensor, states, nr_states
        )
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        return batched_data

    def _encode(
        self,
        env: PlanningEnvironment,
        encoder: GraphEncoderBase,
        atom_values_dicts: list[dict[str, float]],
        atom_values_tensor: torch.Tensor,
        states: Sequence[XState],
        nr_states: int,
    ):
        instance = env.active_instances[0]
        transitions = env.get_applicable_transitions(
            states, instances=[instance] * len(states)
        )
        iterator = zip(
            states,
            atom_values_dicts,
            atom_values_tensor,
            transitions,
        )
        atom_to_index_map = self.atom_to_index_map
        if self.show_progress:
            iterator = tqdm(iterator, total=nr_states, desc="Encoding states")
        batched_data = dict()
        for (
            state,
            state_atom_values_dict,
            state_atom_values_tensor,
            state_transitions,
        ) in iterator:
            data = encoder.to_pyg_data(
                encoder.encode(state.atoms(with_statics=True))
            )  # do not encode the goal!
            reward, done = env.get_reward_and_done(
                state_transitions, instances=[instance] * len(state_transitions)
            )
            data.reward = reward
            data.idx = state.index
            data.done = done
            data.state_desc = str(state)
            data.object_count = len(gather_objects(state))
            atom_str_keys = {str(a) for a in state_atom_values_dict.keys()}
            if missing_atoms := atom_to_index_map.keys() - atom_str_keys:
                # If some atoms are not present in the state, set their values to the correct +- inf
                if self.atom_value_method == AtomValueMethod.MESSAGE_PASSING:
                    default = self.mp_module.init_reward
                else:
                    default = float("inf")
                # convert XAtom keys to str
                values_dict = {str(a): v for a, v in state_atom_values_dict.items()}
                for atom in missing_atoms:
                    values_dict[atom] = default
            else:
                values_dict = state_atom_values_dict
            data.atom_values_dict = values_dict
            data.atom_values_tensor = state_atom_values_tensor
            batched_data[state.index] = data
        return sorted(batched_data.values(), key=lambda d: d.idx)

    def atom_tensor_to_dict(self, atom_tensor: torch.Tensor) -> list[dict[str, float]]:
        return atom_value_tensor_to_dict(
            atom_tensor, self.atom_to_index_map, self.index_to_atom_map
        )

    def atom_value_dict_to_tensor(
        self,
        atom_values: Mapping[XState | int, dict[str | XAtom, float]],
    ) -> torch.Tensor:
        return atom_value_dict_to_tensor(atom_values, self.atom_to_index_map)

    def _compute_atom_dists(
        self,
        env: ExpandedStateSpaceEnv,
        states: List[XState] | None = None,
        atoms: list[XAtom] | None = None,
    ) -> dict[int, defaultdict[XAtom, float]] | torch.Tensor:
        match self.atom_value_method:
            case AtomValueMethod.MESSAGE_PASSING:
                mp_module = OptimalAtomValuesMP(
                    gamma=env.reward_function.gamma,
                    atom_to_index_map=self.atom_to_index_map,
                    aggr="max",
                )
                self.mp_module = mp_module
                pyg_data = self._make_atom_pyg_env(env, atoms=atoms)
                pyg_data.atoms_per_state = [
                    list(state.atoms(with_statics=False)) for state in states
                ]
                values = mp_module(pyg_data)
                del pyg_data.atoms_per_state
                return values
            case AtomValueMethod.DISTANCE_BASED:
                space = env.spaces[0]
                open_list = states or list(space)
                atom_dists: dict[int, defaultdict[XAtom, float]] = dict()
                for state in open_list:
                    state_atom_dists = defaultdict(lambda: float("inf"))
                    for atom in state.atoms(with_statics=False):
                        state_atom_dists[atom] = 0
                    atom_dists[state.index] = state_atom_dists
                broadcast_step = 0
                if self.show_progress:
                    open_list = tqdm(open_list, desc=f"Broadcast Step {broadcast_step}")
                while open_list:
                    new_open_list = self._broadcast_distances(
                        space, atom_dists, open_list
                    )
                    broadcast_step += 1
                    open_list = (
                        tqdm(new_open_list, desc=f"Broadcast Step {broadcast_step}")
                        if self.show_progress
                        else new_open_list
                    )
                return atom_dists
            case _:
                raise ValueError(
                    f"Unsupported atom value method: {self.atom_value_method}"
                )

    def env_aux_data(self) -> dict:
        base_data = super().env_aux_data()
        env = self.env
        if (data := self.try_get_data("aux.atom_to_index_map")) is not None:
            self._atom_to_index_map = data
        if (data := self.try_get_data("aux.pyg_atom_data")) is not None:
            pyg_atom_data = data
            if pyg_atom_data is not None:
                return base_data | dict(
                    pyg_atom_data=pyg_atom_data,
                    atom_to_index_map=self.atom_to_index_map,
                )
        logger = self._get_logger()
        logger.info(f"Auxiliary Data ({AtomDrive.__name__}): Starting.")
        pyg_atom_data = self._make_atom_pyg_env(env)
        logger.info("Auxiliary Data in {AtomDrive.__name__}: Finished.")
        aux_data = base_data | dict(
            pyg_atom_data=pyg_atom_data,
            atom_to_index_map=self.atom_to_index_map,
        )
        self._save_aux_data(
            atom_to_index_map=self.atom_to_index_map,
            pyg_atom_data=pyg_atom_data,
        )
        return aux_data

    def _make_atom_pyg_env(self, env, atoms: Iterable[XAtom] | None = None) -> Data:
        if aux_data := self.try_get_data("aux.pyg_atom_data"):
            return aux_data["aux.pyg_atom_data"]
        space = env.spaces[0]
        self.problem = space.problem
        atoi = self.atom_to_index_map
        if not atoi:
            counter = itertools.count()
            atoi = KeyAwareDefaultDict(lambda atom_str: next(counter))
            self._atom_to_index_map = atoi
        sorted_atoms = sorted(
            atoms
            or chain(
                self.problem.all_atoms(XCategory.fluent),
                self.problem.all_atoms(XCategory.derived),
            ),
            key=lambda a: atoi[str(a)],
        )
        atom_problems = tuple(
            CustomProblem(
                self.problem,
                goal=(XLiteral.make_hollow(atom, False),),
            )
            for atom in sorted_atoms
        )
        return env.to_pyg_data(
            0,
            natural_transitions=True,
            problems=atom_problems,
        )

    @staticmethod
    def _broadcast_distances(
        space: XStateSpace,
        dists: dict[int, MutableMapping[XAtom, float]],
        open_list: Iterable[XState],
    ) -> set[XState]:
        new_open_list = set()
        for state in open_list:
            for pred in space.backward_transitions(state):
                pred_dists = dists[pred.source.index]
                for atom, curr_atom_dist in dists[state.index].items():
                    pred_atom_dist = pred_dists[atom]
                    curr_atom_dist_incr = curr_atom_dist + 1
                    if pred_atom_dist > curr_atom_dist_incr:
                        pred_dists[atom] = curr_atom_dist_incr
                        new_open_list.add(pred.source)
        return new_open_list


class PartialAtomDrive(AtomDrive):
    """
    AtomDrive stores for each state s their optimal distance to any atom p, i.e., a value function V(p | s).

    This drive is used for partial atom spaces, where not all states are present in the state space.
    """

    def __init__(
        self,
        *args,
        iw_search: IWSearch,
        num_states: int = 0,
        seed: int = None,
        max_processes: int | str = 1,
        **kwargs,
    ):
        self.iw_search = iw_search
        self.num_states = num_states
        self.seed = seed
        self._rng = None
        match max_processes:
            case "auto":
                max_processes = env_aware_cpu_count()
            case "all":
                max_processes = env_aware_cpu_count()
            case int():
                max_processes = min(
                    max_processes, env_aware_cpu_count(), self.num_states
                )
            case _:
                raise ValueError(
                    f"Invalid max_processes value: {max_processes}. "
                    "Must be 'auto', 'all', or an integer."
                )
        self._max_processes = max_processes
        kwargs["env"] = None
        kwargs["atom_value_method"] = AtomValueMethod.IW
        super().__init__(*args, **kwargs)

    @property
    def metadata(self) -> dict:
        return dict(
            seed=self.seed,
            num_states=self.num_states,
            expansion_iw_search=self.iw_search,
            **super().metadata,
        )

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    def get_space(self):
        return None

    def env_aux_data(self):
        return None

    @property
    def env(self) -> PlanningEnvironment | ExpandedStateSpaceEnv | None:
        if self._env is not None:
            return self._env
        domain, problem = parse(self.domain_path, self.problem_path)
        successor_gen = XSuccessorGenerator(problem)
        env = SuccessorEnvironment(
            generators=[successor_gen],
            reward_function=self.reward_function,
            reset=True,
            batch_size=1,
        )
        self._env = env
        return env

    def _metadata_misaligned(self, meta: dict) -> str:
        if not super()._metadata_misaligned(meta):
            if (
                self.iw_search is not None
                and self.iw_search != meta["expansion_iw_search"]
            ):
                return f"expansion_iw_search: given={self.iw_search} != loaded={meta['expansion_iw_search']}"
            if self.num_states != meta["num_states"]:
                return f"num_states: given={self.num_states} != loaded={meta['num_states']}"
            if self.seed != meta["seed"]:
                return f"seed: given={self.seed} != loaded={meta['seed']}"
        return ""

    def _build(
        self,
    ) -> List[HeteroData]:
        env = self.env
        problem = env.active_instances[0].problem
        self.problem = problem
        encoder = self.encoder_factory(problem.domain)
        logger = self._get_logger()
        logger.info(
            f"Building {self.__class__.__name__} "
            f"(problem: {problem.name} / {Path(problem.filepath).stem}, #space: {self.num_states} states (max-cutoff))"
        )
        atom_values: dict[XState, dict[str, float]] = self._sample_atom_dists(env)
        states = list(atom_values.keys())
        atom_id_map, atoms = make_atom_ids(problem)
        counter = itertools.count(max(atom_id_map.values()) + 1)
        for state, atom_value_dict in atom_values.items():
            for atom, value in atom_value_dict.items():
                if atom not in atom_id_map:
                    atom_id_map[atom] = next(counter)
        self._atom_to_index_map = atom_id_map
        atom_values_dict = list(atom_values.values())
        atom_values_tensor = atom_value_dict_to_tensor(atom_values, atom_id_map)
        batched_data = self._encode(
            env, encoder, atom_values_dict, atom_values_tensor, states, self.num_states
        )
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {problem.name} / {Path(problem.filepath).stem}, #space: {self.num_states} states (max-cutoff))"
        )
        return batched_data

    def _sample_atom_dists(
        self,
        env: PlanningEnvironment,
    ) -> dict[XState, defaultdict[str, float]]:
        from concurrent.futures import FIRST_COMPLETED, CancelledError, wait

        succ_gen: XSuccessorGenerator = env.active_instances[0]
        dists: dict[XState, defaultdict[str, float]] = dict()
        closed_states = set()
        # progress bar context
        pbar_ctx = (
            tqdm(desc="Sampling Atom Distances", total=self.num_states)
            if self.show_progress
            else contextlib.nullcontext()
        )
        if self._max_processes > 1:
            pbar: tqdm_asyncio | None
            stop_flag = RawValue("b", False)  # a single C‐char, no lock
            init_args = (
                stop_flag,
                self.problem.domain.filepath,
                self.problem.filepath,
                self.iw_search,
            )
            with ProcessPoolExecutor(
                max_workers=self._max_processes,
                initargs=init_args,
                initializer=init_worker,
            ) as pool:
                with pbar_ctx as pbar:
                    # initial dispatch
                    high_seed_end = 2**31 - 1
                    futures = set()
                    for _ in range(min(self._max_processes, self.num_states)):
                        futures.add(
                            pool.submit(
                                _compute_state_dist_mp,
                                int(self.rng.integers(low=0, high=high_seed_end)),
                            )
                        )
                    completed_states = 0
                    while completed_states < self.num_states:
                        done_fs, undone_fs = wait(futures, return_when=FIRST_COMPLETED)
                        futures = undone_fs
                        while completed_states < self.num_states and done_fs:
                            fut = done_fs.pop()
                            if fut.cancelled():
                                continue
                            try:
                                (
                                    init_state_schemas,
                                    init_state_object_bindings,
                                    atom_dists,
                                ) = fut.result()
                            except CancelledError:
                                continue
                            # reconstruct state from serialized action trace
                            state = rebuild_state(
                                succ_gen, init_state_schemas, init_state_object_bindings
                            )
                            if state is None:
                                state = succ_gen.initial_state
                            if state not in closed_states:
                                if pbar is not None:
                                    pbar.update(1)
                                closed_states.add(state)
                                dists[state] = atom_dists
                                completed_states += 1
                                if completed_states == self.num_states:
                                    stop_flag.value = True
                                    break
                            # sample the steps of steps we walk from the start to build the new starting state
                            steps = int(self.rng.random() * len(init_state_schemas))
                            # submit another task to compute the state distance
                            futures.add(
                                pool.submit(
                                    _compute_state_dist_mp,
                                    int(self.rng.integers(low=0, high=high_seed_end)),
                                    True,  # serialize starting state
                                    init_state_schemas[:steps],
                                    init_state_object_bindings[:steps],
                                )
                            )

            return dists
        else:
            # single process case
            state = succ_gen.initial_state
            with pbar_ctx as pbar:
                for _ in range(self.num_states):
                    _, atom_dists = compute_state_dist(
                        state=state,
                        succ_gen=succ_gen,
                        action_schemas=None,
                        object_bindings=None,
                        iw_search=self.iw_search,
                        serialize_starting_state=False,
                        rng=self.rng,
                    )
                    dists[state] = atom_dists
                    if state not in closed_states:
                        closed_states.add(state)
                    if pbar is not None:
                        pbar.update(1)
            return dists


def init_worker(stop_flag, dom_path: str, prob_path: str, iw_search):
    global _domain, _problem, _iw_search, _stop_flag
    _domain, _problem = parse(dom_path, prob_path)
    _iw_search = iw_search
    _stop_flag = stop_flag


def walk(
    successor_gen: XSuccessorGenerator,
    start_state=None,
    rng: np.random.Generator | None = None,
    bounds: tuple[int, int] = (1, 20),
) -> tuple[XState, list[xmi.XAction]]:
    """
    Walks through the environment to collect some states.
    """
    if rng is None:
        rng = globals().get("rng")
        assert rng is not None, "No random number generator provided."
    action_generator = successor_gen.action_generator
    state = successor_gen.initial_state if start_state is None else start_state
    nr_steps = rng.integers(*bounds)
    chosen_actions = []
    for _ in range(nr_steps):
        actions = tuple(action_generator.generate_actions(state))
        if not actions:
            break
        action = rng.choice(actions)
        chosen_actions.append(action)
        state = successor_gen.successor(state, action)
    return state, chosen_actions


def _compute_state_dist_mp(
    seed: int | None = None,
    serialize_starting_state: bool = True,
    action_schemas: Sequence[int] | None = None,
    object_bindings: Sequence[Sequence[str]] | None = None,
):
    global _iw_search, _problem, _stop_flag
    succ_gen = XSuccessorGenerator(_problem)
    return compute_state_dist(
        state=rebuild_state(succ_gen, action_schemas, object_bindings),
        succ_gen=succ_gen,
        action_schemas=action_schemas,
        object_bindings=object_bindings,
        iw_search=_iw_search,
        serialize_starting_state=serialize_starting_state,
        rng=np.random.default_rng(seed),
        stop_flag=_stop_flag,
    )


def compute_state_dist(
    state: XState,
    succ_gen: XSuccessorGenerator,
    action_schemas: Sequence[int] | None = None,
    object_bindings: Sequence[Sequence[str]] | None = None,
    iw_search: IWSearch | None = None,
    serialize_starting_state: bool = True,
    rng: np.random.Generator | None = None,
    stop_flag: Any | None = None,
    bounds: tuple[int, int] = (0, 20),
) -> (
    tuple[Sequence[int], Sequence[str], dict[str, float]]
    | tuple[XState, dict[str, float]]
    | tuple[None, None]
):
    succ_gen = succ_gen
    deadend = True
    start_state = state
    while deadend:
        start_state, action_trace = walk(
            succ_gen,
            start_state=state,
            rng=rng,
            bounds=bounds,
        )
        if any(True for _ in succ_gen.action_generator.generate_actions(start_state)):
            deadend = False
            bounds = (bounds[0] + 2, bounds[1] + 2)
    atom_dists = defaultdict(lambda: float("-inf"))
    collector = CollectorHook()
    if stop_flag is not None:
        expansion_budget = lambda i: not stop_flag.value
    else:
        expansion_budget = return_true
    iw_search.solve(
        succ_gen,
        start_state=start_state,
        novel_hook=collector,
        stop_on_goal=False,
        expansion_budget=expansion_budget,
    )
    if stop_flag.value:
        return None, None  # we will not need this output anymore
    for node in collector.nodes:
        for atom_tuples in node.novelty_trace[-1]:
            if len(atom_tuples) == 1:
                atom = atom_tuples[0]
                reward = -1 * node.depth
                atom_dists[str(atom)] = reward
    if serialize_starting_state:
        # serialized actions will allow to reconstruct the starting state from which IW was started
        serialized_actions = list(
            chain(
                zip(action_schemas or [], object_bindings or []),
                (_serialized_action(action) for action in action_trace),
            )
        )
        schemas, bindings = zip(*serialized_actions) if serialized_actions else ([], [])
        # atom dists will store the reward (negative distance) to each atom in the form of
        #   (predicate_name, (object1_name, object2_name, ...)) --> reward
        return schemas, bindings, dict(atom_dists)
    else:
        return start_state, dict(atom_dists)


def _serialized_atom(atom: XAtom) -> tuple[str, tuple[str, ...]]:
    return str(atom.predicate), tuple(o.get_name() for o in atom.objects)


def _serialized_action(action: xmi.XAction) -> tuple[int, tuple[str, ...]]:
    return action.action_schema.index, tuple(o.get_name() for o in action.objects)


def rebuild_state(
    succ_gen: XSuccessorGenerator,
    action_schemas: Sequence[int],
    object_bindings: Sequence[Sequence[str]],
) -> XState | None:
    if (
        action_schemas is not None
        and action_schemas
        and object_bindings is not None
        and object_bindings
    ):
        actual_schemas = [
            succ_gen.problem.domain.actions[sid] for sid in action_schemas
        ]
        objects = succ_gen.problem.objects
        object_names = list(map(lambda o: o.get_name(), objects))
        bound_objects = [
            [objects[object_names.index(o)] for o in binding]
            for binding in object_bindings
        ]
        action_gen = succ_gen.action_generator
        regrounded_actions = [
            action_gen.ground_action(schema, objs)
            for schema, objs in zip(actual_schemas, bound_objects)
        ]
        rebuilt_state = succ_gen.initial_state
        for action in regrounded_actions:
            rebuilt_state = succ_gen.successor(rebuilt_state, action)
        return rebuilt_state
    return None
