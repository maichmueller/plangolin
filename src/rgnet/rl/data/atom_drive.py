from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Iterable, List, MutableMapping, Union

from torch_geometric.data import Data, HeteroData

import xmimir as xmi
from rgnet.logging_setup import tqdm
from rgnet.rl.envs import ExpandedStateSpaceEnv
from xmimir import XAtom, XCategory, XState, XStateSpace

from .drive import GenericDrive


def make_atom_ids(space: XStateSpace) -> dict[str, int]:
    nr_fluent_atoms = space.atom_count(XCategory.fluent)
    factor, remainder = divmod(nr_fluent_atoms, 10)
    if remainder > 0:
        factor += 1
    atom_lookup = {}
    for atom in space.all_atoms(XCategory.fluent):
        atom_lookup[str(atom)] = atom.base.get_index()
    for atom in space.all_atoms(XCategory.derived):
        atom_lookup[str(atom)] = 10**factor + atom.base.get_index()
    return atom_lookup


class AtomDrive(GenericDrive):
    """
    AtomDrive stores for each state s their optimal distance to any atom p, i.e., a value function V(p | s).
    """

    def __init__(self, *args, **kwargs):
        self._atom_to_index_map: dict[str, int] = None
        self._index_to_atom_map: dict[int, str] = None
        super().__init__(*args, **kwargs)

    @cached_property
    def atom_to_index_map(self):
        if self._atom_to_index_map is None:
            self._atom_to_index_map = make_atom_ids(super()._make_space())
        return self._atom_to_index_map

    @cached_property
    def index_to_atom_map(self):
        if self._index_to_atom_map is None:
            self._index_to_atom_map = {v: k for k, v in self.atom_to_index_map.items()}
        return self._index_to_atom_map

    def _build(
        self,
        env: ExpandedStateSpaceEnv,
    ) -> List[HeteroData]:
        space: xmi.XStateSpace = env.active_instances[0]
        encoder = self.encoder_factory(space.problem.domain)
        self.desc = f"{self.__class__.__name__}({space.problem.name}, {space.problem.filepath}, state_space={str(space)})"
        nr_states: int = len(space)
        logger = self._get_logger()
        logger.info(
            f"Building {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        atom_ids = make_atom_ids(space)
        self._atom_to_index_map = atom_ids
        # Each data object represents one state
        batched_data: List[Union[HeteroData, Data]] = [None] * nr_states
        states = list(space)
        open_list = states
        atom_dists: list[defaultdict[XAtom, float]] = [None] * nr_states
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

        transitions = env.get_applicable_transitions(states)
        iterator = zip(states, atom_dists, transitions)
        if self.show_progress:
            iterator = tqdm(iterator, total=nr_states, desc="Encoding states")
        for state, state_atom_dists, state_transitions in iterator:
            data = encoder.to_pyg_data(encoder.encode(state))
            reward, done = env.get_reward_and_done(
                state_transitions, instances=[space] * len(state_transitions)
            )
            data.reward = reward
            data.idx = state.index
            data.done = done
            data.atom_distances = {
                str(atom): dist for atom, dist in state_atom_dists.items()
            }

            #     {atom_ids[str(atom)]: dist for atom, dist in state_atom_dists.items()},
            #     batch_size=1,
            # )
            batched_data[state.index] = data
        logger.info(
            f"Finished {self.__class__.__name__} "
            f"(problem: {space.problem.name} / {Path(space.problem.filepath).stem}, #space: {space})"
        )
        return batched_data

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
