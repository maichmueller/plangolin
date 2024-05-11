import logging
import pathlib
import re
from typing import Callable, Dict, List, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pymimir as mi
import torch
import torch_geometric as pyg
from pymimir import Action, State
from torch import Tensor

from experiments.data_layout import DataLayout, DatasetType
from rgnet.encoding import HeteroGraphEncoder
from rgnet.models import LightningHetero
from rgnet.supervised import MultiInstanceSupervisedSet


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


class CompletedExperiment:

    def __init__(self, data_layout: DataLayout, run_id: str, device: str = "cpu"):
        self.data_layout = data_layout

        assert data_layout.encoder_type == "hetero"  # TODO allow other encoder

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
        L.Trainer().validate(self.model, loader)
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
            return self.model(data.x_dict, data.edge_index_dict)

        return value_function

    def run_policy(self, dataset_type: DatasetType, max_steps=100):
        """
        1. Parse optimal plans.
        2. Run ValuePolicy using the trained model on each problem of dataset_type.
        :param dataset_type: Which problems to use.
        :return: A List of found plans (list of actions) or None if no plan was found.
        """
        plans = self.data_layout.plans_paths_for(dataset_type)
        policy = ValuePolicy(self.wrap_model_as_value_function())
        found_plans = []
        with torch.no_grad():
            self.model.eval()
            for plan_file in plans:
                problem_path = self.data_layout.problem_for_plan(
                    plan_file, dataset_type
                )
                problem = mi.ProblemParser(str(problem_path.absolute())).parse(
                    self.domain
                )
                opt_plan, opt_cost = parse_plan(plan_file, problem)
                result = policy.run(problem, max_steps)
                if result is None:
                    found_plans.append(None)
                    logging.info(f"Could not find a plan after {max_steps} steps.")
                    continue
                plan, cost = result
                logging.info(f"Found plan with cost {cost}, optimal cost is {opt_cost}")
                found_plans.append(plan)
        return found_plans


class ValuePolicy:

    def __init__(self, value_function: Callable[[mi.State], float]):
        self.value_function = value_function

    def __call__(
        self, state: mi.State, succ: mi.GroundedSuccessorGenerator
    ) -> Tuple[mi.Action, mi.State]:
        actions = succ.get_applicable_actions(state)
        action_targets = [(a, a.apply(state)) for a in actions]
        return min(action_targets, key=lambda a_t: self.value_function(a_t[1]))

    def evaluate_actions(self, state: mi.State, succ: mi.SuccessorGenerator):
        """Return a list containing (Action, next-state, value(next-state) tuples.
        Sorted by the value (smallest to highest)
        """
        actions = succ.get_applicable_actions(state)
        action_targets: List[Tuple[Action, State]] = [
            (a, a.apply(state)) for a in actions
        ]
        a_t_v = [
            (action, next_state, self.value_function(next_state))
            for action, next_state in action_targets
        ]
        return sorted(a_t_v, key=lambda atv: atv[2])

    def run(self, problem: mi.Problem, max_steps: int):

        plan = []
        steps = 0
        succ = mi.GroundedSuccessorGenerator(problem)
        state = problem.create_state(problem.initial)
        while not state.literals_hold(problem.goal) and steps <= max_steps:
            action, next_state = self(state, succ)
            logging.debug(f"Selected {action} in state {state}")
            plan.append(action)
            state = next_state
            steps += 1
        if not state.literals_hold(problem.goal):
            logging.info(f"Could not find a plan in {max_steps}.")
            return None
        plan_cost = sum(a.cost for a in plan)
        return plan, plan_cost
