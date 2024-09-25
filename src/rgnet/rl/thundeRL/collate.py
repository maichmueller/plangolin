import itertools

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from rgnet.rl.agents import ActorCritic
from rgnet.rl.envs import PlanningEnvironment


class Collate:

    def __init__(
        self,
        # env_keys: PlanningEnvironment.AcceptedKeys,
        # actor_keys: ActorCritic.AcceptedKeys,
    ):
        super().__init__()

    def __call__(self, data_list, **kwargs):
        flattened_targets = list(
            itertools.chain.from_iterable([d.targets for d in data_list])
        )

        successor_batch = Batch.from_data_list(flattened_targets)

        batched = Batch.from_data_list(data_list, exclude_keys=["targets"])

        num_successors = torch.tensor(
            [data.reward.numel() for data in data_list], dtype=torch.long
        )

        return batched, successor_batch, num_successors
