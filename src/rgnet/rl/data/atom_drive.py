import contextlib
import itertools
from collections import defaultdict, deque
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from tqdm.asyncio import tqdm_asyncio

import xmimir as xmi
from rgnet.algorithms.policy_evaluation_mp import OptimalAtomValuesMP
from rgnet.logging_setup import tqdm
from rgnet.rl.envs import (
    ExpandedStateSpaceEnv,
    PlanningEnvironment,
    SuccessorEnvironment,
)
from rgnet.utils.misc import KeyAwareDefaultDict
from xmimir import XAtom, XCategory, XState, XStateSpace, XSuccessorGenerator, parse
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


def atom_tensor_to_dict(
    atom_tensor: torch.Tensor,
    atom_to_index_map: Mapping[str, int],
    index_to_atom_map: Mapping[int, str],
) -> list[dict[str, float]]:
    atom_ids = torch.tensor(list(atom_to_index_map.values()), dtype=torch.int)
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
        self._atom_to_index_map: dict[str, int] = None
        self._index_to_atom_map: dict[int, str] = None
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
        if self.store_values_as_tensor:
            if isinstance(atom_values, dict):
                atom_values = self.atom_value_dict_to_tensor(atom_values)
        else:
            if isinstance(atom_values, torch.Tensor):
                atom_values = self.atom_tensor_to_dict(atom_values)
        batched_data = self._encode(env, encoder, atom_values, states, nr_states)
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        return batched_data

    def _encode(self, env, encoder, atom_values, states, nr_states):
        instance = env.active_instances[0]
        transitions = env.get_applicable_transitions(
            states, instances=[instance] * len(states)
        )
        iterator = zip(states, atom_values, transitions)
        if self.show_progress:
            iterator = tqdm(iterator, total=nr_states, desc="Encoding states")
        batched_data = dict()
        for state, state_atom_values, state_transitions in iterator:
            data = encoder.to_pyg_data(encoder.encode(state))
            reward, done = env.get_reward_and_done(
                state_transitions, instances=[instance] * len(state_transitions)
            )
            data.reward = reward
            data.idx = state.index
            data.done = done
            data.atom_values = state_atom_values
            batched_data[state.index] = data
        return sorted(batched_data.values(), key=lambda d: d.idx)

    def atom_tensor_to_dict(self, atom_tensor: torch.Tensor) -> list[dict[str, float]]:
        return atom_tensor_to_dict(
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
        if self.try_get_data("aux.atom_to_index_map") is not None:
            self._atom_to_index_map = self.try_get_data("aux.atom_to_index_map")
        if self.try_get_data("aux.pyg_atom_data") is not None:
            pyg_atom_data = self.try_get_data("aux.pyg_atom_data")
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
        atom_value_method: bool = False,
        store_values_as_tensor: bool = True,
        seed: int = None,
        **kwargs,
    ):
        self._atom_to_index_map: dict[str, int] = None
        self._index_to_atom_map: dict[int, str] = None
        self.atom_value_method = atom_value_method
        self.store_values_as_tensor = store_values_as_tensor
        self.iw_search = iw_search
        self.num_states = num_states
        self.seed = seed
        kwargs["env"] = None
        super().__init__(*args, **kwargs)

    @property
    def metadata(self) -> dict:
        return dict(
            seed=self.seed,
            num_states=self.num_states,
            expansion_iw_search=self.iw_search,
            **super().metadata,
        )

    def get_space(self):
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
        atom_values = self._sample_atom_dists(env)
        nr_states: int = self.num_states
        atom_id_map, atoms = make_atom_ids(problem)
        self._atom_to_index_map = atom_id_map
        states = list(atom_values.keys())
        if self.store_values_as_tensor:
            if isinstance(atom_values, dict):
                atom_values = atom_value_dict_to_tensor(
                    atom_values, self.atom_to_index_map
                )
        else:
            if isinstance(atom_values, torch.Tensor):
                atom_values = atom_tensor_to_dict(
                    atom_values, self.atom_to_index_map, self.index_to_atom_map
                )

        batched_data = self._encode(env, encoder, atom_values, states, nr_states)
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {problem.name} / {Path(problem.filepath).stem}, #space: {self.num_states} states (max-cutoff))"
        )
        return batched_data

    def _sample_atom_dists(
        self,
        env: SuccessorEnvironment,
    ) -> dict[XState, defaultdict[XAtom, float]] | torch.Tensor:
        succ_gen = env.active_instances[0]
        rng = np.random.default_rng(self.seed)
        open_states = deque([succ_gen.initial_state])
        dists: dict[XState, defaultdict[XAtom, float]] = dict()
        closed_states = set()
        pbar: tqdm_asyncio | None
        with (
            tqdm(desc="Sampling Atom Distances", total=self.num_states)
            if self.show_progress
            else contextlib.nullcontext()
        ) as pbar:
            while len(closed_states) < self.num_states or not open_states:
                state = open_states.pop()
                closed_states.add(state)
                atom_dists = defaultdict(lambda: float("inf"))
                dists[state] = atom_dists
                collector = CollectorHook()
                self.iw_search.solve(
                    succ_gen,
                    start_state=state,
                    novel_hook=collector,
                    stop_on_goal=False,
                )
                nodes = collector.nodes
                for node in (nodes[i] for i in rng.permutation(len(nodes))):
                    if node.state not in closed_states:
                        open_states.append(node.state)
                    for atom_tuples in node.novelty_trace[-1]:
                        for atom in atom_tuples:
                            if len(atom_tuples) == 1:
                                atom_dists[atom] = node.depth
                if pbar:
                    pbar.update(1)
        return dists

    def env_aux_data(self):
        return None
