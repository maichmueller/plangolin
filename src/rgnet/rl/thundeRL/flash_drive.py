import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pymimir as mi
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset

from rgnet import HeteroGraphEncoder
from rgnet.rl.envs import ExpandedStateSpaceEnv, MultiInstanceStateSpaceEnv
from rgnet.rl.envs.expanded_state_space_env import IteratingReset


def build_from_env(
    env: MultiInstanceStateSpaceEnv, num_batches: int, encoder: HeteroGraphEncoder
) -> List[HeteroData]:
    state_to_idx: Dict[mi.StateSpace, Dict[mi.State, int]] = dict()

    def query():
        out = env.reset()
        instances: List[mi.StateSpace] = out[env.keys.instance]
        for space in instances:
            if space not in state_to_idx:
                state_to_idx[space] = {s: i for i, s in enumerate(space.get_states())}

        states: List[mi.State] = out[env.keys.state]
        batched_transitions: List[List[mi.Transition]] = out[env.keys.transitions]
        batched_data: List[HeteroData] = [
            encoder.to_pyg_data(encoder.encode(state)) for state in states
        ]

        # Each data object represents one state
        # It the index of all neighboring states, the done and reward signals
        for i, data in enumerate(batched_data):
            reward, done = env.get_reward_and_done(
                actions=batched_transitions[i],
                current_states=[t.source for t in batched_transitions[i]],
                instances=[instances[i]] * len(batched_transitions[i]),
            )
            data.reward = reward
            # Save the index of the state
            # NOTE: No element should contain index
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html
            data.idx = state_to_idx[instances[i]][states[i]]
            data.done = done
            # List[HData] the encoded targets of the transitions
            data.targets = [
                encoder.to_pyg_data(encoder.encode(transition.target))
                for transition in batched_transitions[i]
            ]
        return batched_data

    return list(itertools.chain.from_iterable(query() for _ in range(num_batches)))


class FlashDrive(InMemoryDataset):

    def __init__(
        self,
        domain_path: Path,
        problem_path: Path,
        custom_dead_enc_reward: float,
        max_expanded: Optional[int] = None,
        root_dir: Optional[str] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        assert domain_path.exists() and domain_path.is_file()
        assert problem_path.exists() and problem_path.is_file()
        self.domain_file: Path = domain_path
        self.problem_path: Path = problem_path
        self.custom_dead_enc_reward = custom_dead_enc_reward
        self.max_expanded = max_expanded
        super().__init__(
            root=root_dir,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            log=log,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [self.problem_path.stem + ".pt"]

    def process(self) -> None:
        domain = mi.DomainParser(str(self.domain_file.absolute())).parse()
        problem = mi.ProblemParser(str(self.problem_path.absolute())).parse(domain)
        space = mi.StateSpace.new(
            problem,
            mi.GroundedSuccessorGenerator(problem),
            self.max_expanded or 1_000_000,
        )
        env = ExpandedStateSpaceEnv(
            space,
            batch_size=torch.Size((space.num_states(),)),
            reset_strategy=IteratingReset(),
            custom_dead_end_reward=self.custom_dead_enc_reward,
        )
        data_list = build_from_env(env, 1, HeteroGraphEncoder(domain))
        self.save(data_list, self.processed_paths[0])

    def __getattr__(self, key: str) -> Any:
        """InMemoryDataset forgot the poor HeteroData objects, logic is equivalent."""
        data = self.__dict__.get("_data")
        if isinstance(data, HeteroData) and key in data:
            if self._indices is None and data.__inc__(key, data[key]) == 0:
                return data[key]
            else:
                data_list = [self.get(i) for i in self.indices()]
                return Batch.from_data_list(data_list)[key]
        else:
            return super().__getattr__(key)
