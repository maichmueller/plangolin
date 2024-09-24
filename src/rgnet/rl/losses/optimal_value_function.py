from typing import Dict, List

import pymimir as mi
import torch
from tensordict import NestedKey
from torchrl.modules import ValueOperator

from rgnet.rl.non_tensor_data_utils import NonTensorWrapper, non_tensor_to_list


class OptimalValueFunction(torch.nn.Module):
    """Don't predict the value target just use the discounted distance to goal"""

    def __init__(self, optimal_values: Dict[mi.State, float], device: torch.device):
        super().__init__()
        self.optimal_values = optimal_values
        self.device = device

    def __call__(
        self, batched_states: List[mi.State] | NonTensorWrapper
    ) -> torch.Tensor:
        batched_states = non_tensor_to_list(batched_states)
        return torch.stack(
            [
                torch.tensor(
                    [self.optimal_values[state] for state in states],
                    dtype=torch.float,
                    device=self.device,
                ).view(-1, 1)
                for states in batched_states
            ]
        )

    def as_td_module(
        self, state_key: NestedKey, state_value_key: NestedKey
    ) -> ValueOperator:
        return ValueOperator(
            module=self, in_keys=[state_key], out_keys=[state_value_key]
        )
