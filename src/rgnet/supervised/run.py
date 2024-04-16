import logging
import os
import time

import torch

from rgnet.encoding import ColorGraphEncoder
from rgnet.pddl_import import import_all_from, import_problems
from rgnet.supervised.data import MultiInstanceSupervisedSet
from rgnet.supervised.training import training, evaluate
from rgnet.utils import get_device_cuda_if_possible


def run():
    curr_dir = os.getcwd()
    logging.getLogger().setLevel(logging.INFO)
    device: torch.device = get_device_cuda_if_possible()
    logging.info(f"Using {device.type} as device")
    logging.info("Working from " + curr_dir)
    start_time = time.time()

    import_time = time.time()
    problem_path = curr_dir + "/data/pddl_domains/blocks"
    domain, problems = import_all_from(problem_path)
    dataset_path = curr_dir + "/data/datasets/blocks"
    training_set = MultiInstanceSupervisedSet(
        problems, ColorGraphEncoder(domain), root=dataset_path + "/training", log=True
    )
    logging.info(f"Took {time.time() - import_time}s to construct the dataset.")
    logging.info(f"Dataset contains {len(training_set)} graphs")
    start_time_training = time.time()
    model = training(training_set, device)
    logging.info(f"Took {time.time() - start_time_training:.2f}s to train the model")

    # Evaluate the model
    evaluation_problems = import_problems(problem_path + "/test", domain)
    evaluation_set = MultiInstanceSupervisedSet(
        evaluation_problems, ColorGraphEncoder(domain), root=dataset_path + "/test"
    )
    evaluate(model, evaluation_set)

    logging.info(f"Completed run after {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    run()
