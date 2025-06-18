import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Iterable, List, MutableMapping, Sequence, Union

import torch
from torch_geometric.data import Data, HeteroData

import xmimir as xmi
from rgnet.algorithms.policy_evaluation_mp import OptimalAtomValuesMP
from rgnet.logging_setup import tqdm
from rgnet.rl.envs import ExpandedStateSpaceEnv
from xmimir import XAtom, XCategory, XState, XStateSpace
from xmimir.wrappers import CustomProblem, XLiteral

from .drive import BaseDrive, BaseEnvAuxData


def make_atom_ids(space: XStateSpace) -> tuple[dict[str, int], list[XAtom]]:
    nr_fluent_atoms = space.atom_count(XCategory.fluent)
    factor, remainder = divmod(nr_fluent_atoms, 10)
    if remainder > 0:
        factor += 1
    atom_lookup = {}
    all_atoms = []
    for atom in space.all_atoms(XCategory.fluent):
        atom_lookup[str(atom)] = atom.base.get_index()
        all_atoms.append(atom)
    for atom in space.all_atoms(XCategory.derived):
        atom_lookup[str(atom)] = 10**factor + atom.base.get_index()
        all_atoms.append(atom)
    return atom_lookup, all_atoms


@dataclass(frozen=True)
class AtomEnvAuxData(BaseEnvAuxData):
    pyg_atom_data: Data


class AtomDrive(BaseDrive):
    """
    AtomDrive stores for each state s their optimal distance to any atom p, i.e., a value function V(p | s).
    """

    def __init__(
        self,
        *args,
        distance_based_atom_values: bool = False,
        store_values_as_tensor: bool = True,
        **kwargs,
    ):
        self._atom_to_index_map: dict[str, int] = None
        self._index_to_atom_map: dict[int, str] = None
        self.distance_based_atom_values = distance_based_atom_values
        self.store_values_as_tensor = store_values_as_tensor
        super().__init__(*args, **kwargs)

    @cached_property
    def atom_to_index_map(self):
        if self._atom_to_index_map is None:
            if self.metadata_path.exists():
                self._atom_to_index_map = self._load_metadata()[-1]
            if self._atom_to_index_map is None:
                self._atom_to_index_map, _ = make_atom_ids(super()._make_space())
        return self._atom_to_index_map

    @cached_property
    def index_to_atom_map(self):
        if self._index_to_atom_map is None:
            self._index_to_atom_map = {v: k for k, v in self.atom_to_index_map.items()}
        return self._index_to_atom_map

    def _save_metadata(self, *extras):
        super()._save_metadata(*(*extras, self.atom_to_index_map))

    def _build(
        self,
        env: ExpandedStateSpaceEnv,
    ) -> List[HeteroData]:
        space: xmi.XStateSpace = env.active_instances[0]
        encoder = self.encoder_factory(space.problem.domain)
        nr_states: int = len(space)
        logger = self._get_logger()
        logger.info(
            f"Building {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        atom_id_map, atoms = make_atom_ids(space)
        self._atom_to_index_map = atom_id_map
        # Each data object represents one state
        batched_data: List[Union[HeteroData, Data]] = [None] * nr_states
        states = list(space)
        atom_values = self._compute_atom_dists(env, states)
        if self.store_values_as_tensor:
            if isinstance(atom_values, list):
                atom_values = self.atom_value_dict_to_tensor(atom_values)
        else:
            if isinstance(atom_values, torch.Tensor):
                atom_values = self.atom_tensor_to_dict(atom_values)

        transitions = env.get_applicable_transitions(
            states, instances=[space] * len(states)
        )
        iterator = zip(states, atom_values, transitions)
        if self.show_progress:
            iterator = tqdm(iterator, total=nr_states, desc="Encoding states")
        for state, state_atom_values, state_transitions in iterator:
            data = encoder.to_pyg_data(encoder.encode(state))
            reward, done = env.get_reward_and_done(
                state_transitions, instances=[space] * len(state_transitions)
            )
            data.reward = reward
            data.idx = state.index
            data.done = done
            data.atom_values = state_atom_values
            batched_data[state.index] = data
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        return batched_data

    def atom_tensor_to_dict(self, atom_tensor: torch.Tensor) -> list[dict[str, float]]:
        atom_id_map = self.atom_to_index_map
        atom_ids = torch.tensor(list(atom_id_map.values()), dtype=torch.int)
        assert (
            atom_ids == torch.arange(atom_ids.min().item(), atom_ids.max().item() + 1)
        ).all()
        atom_values = []
        if atom_tensor.ndim == 1:
            atom_tensor = atom_tensor.unsqueeze(0)
        for state_atom_values in atom_tensor:
            atom_value_dict = {}
            for j, value in enumerate(state_atom_values):
                atom_value_dict[self.index_to_atom_map[j]] = value.item()
            atom_values.append(atom_value_dict)
        return atom_values

    def atom_value_dict_to_tensor(
        self, atom_values: Sequence[dict[XAtom, float]]
    ) -> torch.Tensor:
        atom_id_map = self.atom_to_index_map
        atom_ids = torch.tensor(list(atom_id_map.values()), dtype=torch.int)
        assert (
            atom_ids == torch.arange(atom_ids.min().item(), atom_ids.max().item() + 1)
        ).all()
        atom_tensor = torch.full(
            (len(atom_values), len(atom_ids)), torch.inf, dtype=torch.float
        )
        for i, value_dict in enumerate(atom_values):
            atom_indices = [atom_id_map[str(atom)] for atom in value_dict.keys()]
            atom_tensor[i, atom_indices] = torch.tensor(
                list(value_dict.values()), dtype=torch.float
            )
        atom_values = atom_tensor
        return atom_values

    def _compute_atom_dists(
        self,
        env: ExpandedStateSpaceEnv,
        states: List[XState] | None = None,
        atoms: list[XAtom] | None = None,
    ) -> list[defaultdict[XAtom, float]] | torch.Tensor:
        if not self.distance_based_atom_values:
            mp_module = OptimalAtomValuesMP(
                gamma=env.reward_function.gamma,
                atom_to_index_map=self.atom_to_index_map,
                aggr="max",
            )
            pyg_data = self._make_atom_pyg_env(env, atoms=atoms)
            pyg_data.atoms_per_state = [
                list(state.atoms(with_statics=False)) for state in states
            ]
            return mp_module(pyg_data)
        else:
            space = env.spaces[0]
            open_list = states or list(space)
            atom_dists: list[defaultdict[XAtom, float]] = [None] * len(states)
            for state in open_list:
                state_atom_dists = defaultdict(lambda: float("inf"))
                for atom in state.atoms(with_statics=False):
                    state_atom_dists[atom] = 0
                atom_dists[state.index] = state_atom_dists
            broadcast_step = 0
            if self.show_progress:
                open_list = tqdm(open_list, desc=f"Broadcast Step {broadcast_step}")
            while open_list:
                new_open_list = self._broadcast_distances(space, atom_dists, open_list)
                broadcast_step += 1
                open_list = (
                    tqdm(new_open_list, desc=f"Broadcast Step {broadcast_step}")
                    if self.show_progress
                    else new_open_list
                )
            return atom_dists

    def _make_env_aux_data(self, env: ExpandedStateSpaceEnv) -> AtomEnvAuxData:
        base_data = dataclasses.asdict(super()._make_env_aux_data(env))
        logger = self._get_logger()
        logger.info(f"Auxiliary Data ({AtomDrive.__name__}): Starting.")
        pyg_atom_data = self._make_atom_pyg_env(env)
        logger.info("Auxiliary Data in {AtomDrive.__name__}: Finished.")
        return AtomEnvAuxData(
            **base_data,
            pyg_atom_data=pyg_atom_data,
        )

    def _make_atom_pyg_env(self, env, atoms: Iterable[XAtom] | None = None) -> Data:
        space = env.spaces[0]
        atoi = self.atom_to_index_map
        sorted_atoms = sorted(
            atoms
            or chain(
                space.all_atoms(XCategory.fluent),
                space.all_atoms(XCategory.derived),
            ),
            key=lambda a: atoi[str(a)],
        )
        atom_problems = tuple(
            CustomProblem(
                env.spaces[0].problem,
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
        dists: list[MutableMapping[XAtom, float]],
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
        domain, problem = parse(self.domain_path, self.problem_path)
        successor_gen = XSuccessorGenerator(problem)
        env = SuccessorEnvironment(
            generators=[successor_gen],
            reward_function=self.reward_function,
            reset=True,
            batch_size=1,
        )
        kwargs["env"] = env
        super().__init__(*args, **kwargs)

    @property
    def metadata(self) -> dict:
        return dict(
            seed=self.seed,
            num_states=self.num_states,
            expansion_iw_search=self.iw_search,
            **super().metadata,
        )

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
