import dataclasses
import logging
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pymimir as mi
import torch
import torch_geometric as pyg
from torch import Tensor

from experiments.data_layout import DataLayout, DatasetType
from experiments.policy import Policy, ValuePolicy
from rgnet import LightningHetero
from rgnet.encoding import HeteroGraphEncoder
from rgnet.supervised import MultiInstanceSupervisedSet
from rgnet.utils import get_device_cuda_if_possible


def plot_prediction_for_label(pred_for_label: Dict[int, List[Tensor]], label):
    """Plot the distribution of predicted labels for a given expected label."""
    predictions = Tensor(pred_for_label[label]).numpy()
    values, counts = np.unique(predictions, return_counts=True)
    plt.bar(values, counts)
    plt.xlabel("Predicted label")
    plt.ylabel("Number of samples")
    plt.title(f"Predicted label distribution for expected label {label}")
    plt.show()


def parse_plan(path: pathlib.Path, problem: mi.Problem) -> Tuple[List[mi.Action], int]:
    """
    Tries to parse plan file by matching actions to applicable actions in the problem.
    :param path: Path to the plan file.
    :param problem: The problem for which the plan is valid.
    :return: A tuple containing a list of actions and the cost of the plan.
    """
    assert path.is_file(), path.absolute()
    lines = path.read_text().splitlines()
    succ = mi.GroundedSuccessorGenerator(problem)
    state = problem.create_state(problem.initial)

    # fast-downward stores plans as (action-schema obj1 obj2)
    def format_action(a: mi.Action):
        schema_name = a.schema.name
        obj = [o.name for o in a.get_arguments()]
        return "(" + schema_name + " " + " ".join(obj) + ")"

    action_list = []
    for action_name in lines:
        if not action_name.startswith("("):
            break
        action = next(
            (
                a
                for a in succ.get_applicable_actions(state)
                if format_action(a) == action_name
            ),
            None,
        )
        if action is None:
            raise ValueError(
                "Could not find applicable action for "
                f"{action_name}. Applicable actions are"
                f"{[format_action(a) for a in succ.get_applicable_actions(state)]}."
            )
        action_list.append(action)
        state = action.apply(state)
    cost = re.search(r"cost = (\d+)", lines[-1])
    if cost is None:
        raise ValueError(f"Could not find cost in {lines[-1]}")
    cost = int(cost.group(1))
    assert sum(a.cost for a in action_list) == cost
    return action_list, cost


@dataclasses.dataclass
class PlanResult:
    plan: Optional[List[mi.Action]]
    cost: Optional[int]
    optimal_plan: List[mi.Action]
    opt_cost: int
    problem: str


class CompletedExperiment:

    def __init__(
        self,
        data_layout: DataLayout,
        run_id: str,
        device: torch.device = get_device_cuda_if_possible(),
    ):
        self.data_layout = data_layout

        assert data_layout.encoder_type == "hetero"  # TODO allow other encoder
        self.device: torch.device = device
        self.domain = mi.DomainParser(
            str(data_layout.domain_file_path.absolute())
        ).parse()

        # Load datasets on demand
        self._datasets: Dict[DatasetType, MultiInstanceSupervisedSet] = dict()
        self.encoder = HeteroGraphEncoder(self.domain)
        checkpoints = list(data_layout.load_checkpoints(run_id))
        assert len(checkpoints) > 0, f"No checkpoint found for run {run_id}"
        if len(checkpoints) > 1:
            logging.warning("More than one checkpoint found, using the first one.")
        logging.info(f"Using {device} for model inference.")
        self.model: LightningHetero = LightningHetero.load_from_checkpoint(
            checkpoints[0], map_location=device
        )

    def _get_dataset(self, dataset_type: DatasetType):
        if dataset_type not in self._datasets:
            self._datasets[dataset_type] = self.data_layout.load_dataset(dataset_type)
        return self._datasets[dataset_type]

    @property
    def train_set(self):
        return self._get_dataset(DatasetType.TRAIN)

    @property
    def eval_set(self):
        return self._get_dataset(DatasetType.EVAL)

    @property
    def test_set(self):
        return self._get_dataset(DatasetType.TEST)

    def plot_label_distribution(self, dataset_type: DatasetType):
        label_tensor: Tensor = self._get_dataset(dataset_type).y
        label_distribution = label_tensor.bincount()
        # Plot histogram of loss by label
        plt.bar(torch.arange(len(label_distribution)), label_distribution)

        # Add labels and title
        plt.xlabel("Label")
        plt.ylabel("Number of sample/graphs")
        plt.title(f"Samples per label for {dataset_type} dataset")

        # Show plot
        plt.show()

    def eval_and_plot_loss_by_label(self, dataset_type: DatasetType):
        """Plots the mean loss for each occurring label in the dataset.
        This will evaluate the model on the dataset which might take some time..
        """
        loader = pyg.loader.DataLoader(
            self._get_dataset(dataset_type),
            batch_size=256,
            shuffle=False,
            num_workers=4,
        )
        L.Trainer(accelerator=self.device).validate(self.model, loader)
        loss_distribution = {
            label: torch.tensor(losses).mean()
            for label, losses in self.model.val_loss_by_label.items()
        }
        # Plot histogram of loss by label
        plt.bar(loss_distribution.keys(), loss_distribution.values())

        # Add labels and title
        plt.xlabel("Label")
        plt.ylabel(f"{self.model.loss_function} Loss")
        plt.title(f"Loss per label for {dataset_type} dataset")

        # Show plot
        plt.show()

    def wrap_model_as_value_function(self):
        def value_function(state: mi.State):
            data = self.encoder.to_pyg_data(self.encoder.encode(state))
            data.to(self.model.device)
            return self.model(data.x_dict, data.edge_index_dict)

        return value_function

    def run_vpolicy(self, dataset_type: DatasetType, max_steps=100) -> List[PlanResult]:
        """
        1. Parse optimal plans.
        2. Run ValuePolicy using the trained model on each problem of dataset_type.
        :param dataset_type: Which problems to use.
        :return: A List of found plans (list of actions) or None if no plan was found.
        """
        policy = ValuePolicy(self.wrap_model_as_value_function())
        with torch.no_grad():
            self.model.eval()
            return self.run_policy(dataset_type, policy, max_steps)

    def run_policy(
        self, dataset_type: DatasetType, policy: Policy, max_steps
    ) -> List[PlanResult]:
        plans = self.data_layout.plans_paths_for(dataset_type)
        found_plans: List[PlanResult] = []
        for plan_file in plans:
            problem_path = self.data_layout.problem_for_plan(plan_file, dataset_type)
            problem = mi.ProblemParser(str(problem_path.absolute())).parse(self.domain)
            opt_plan, opt_cost = parse_plan(plan_file, problem)
            result = policy.run(problem, max_steps)
            if result is None:
                found_plans.append(
                    PlanResult(None, None, opt_plan, opt_cost, problem.name)
                )
                logging.info(f"Could not find a plan after {max_steps} steps.")
                continue
            plan, cost = result
            logging.info(f"Found plan with cost {cost}, optimal cost is {opt_cost}")
            found_plans.append(PlanResult(plan, cost, opt_plan, opt_cost, problem.name))
        return found_plans
