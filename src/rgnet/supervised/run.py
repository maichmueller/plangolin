import logging
import os
import pathlib
import time
from datetime import datetime

import torch

from rgnet.encoding import ColorGraphEncoder
from rgnet.pddl_import import import_all_from, import_problems
from rgnet.supervised.data import MultiInstanceSupervisedSet
from rgnet.supervised.training import Trainer
from rgnet.utils import get_device_cuda_if_possible, time_delta_now


def run():
    curr_dir = os.getcwd()
    logging.getLogger().setLevel(logging.INFO)
    device: torch.device = get_device_cuda_if_possible()
    logging.info(f"Using {device.type} as device")
    logging.info("Working from " + curr_dir)
    start_time = time.time()

    # define paths
    data_path = curr_dir + "/data"
    problem_path = data_path + "/pddl_domains/blocks"
    test_problem_path = problem_path + "/test"
    dataset_path = data_path + "/datasets/blocks"
    training_set_path = dataset_path + "/training"
    test_set_path = dataset_path + "/test"
    model_save_path = pathlib.Path(
        data_path + f"/models/run{datetime.now().strftime('%y%m%d_%H%M%S')}.pt"
    )
    model_save_path.parent.mkdir(exist_ok=True)

    import_time = time.time()
    domain, problems = import_all_from(problem_path)
    training_set = MultiInstanceSupervisedSet(
        problems, ColorGraphEncoder(domain), root=training_set_path, log=True
    )
    # Evaluate the model
    evaluation_problems = import_problems(test_problem_path, domain)
    evaluation_set = MultiInstanceSupervisedSet(
        evaluation_problems, ColorGraphEncoder(domain), root=test_set_path
    )
    logging.info(f"Took {time_delta_now(import_time)} to construct the dataset.")

    logging.info(f"Training dataset contains {len(training_set)} graphs/states")
    # logging.info(training_set.get_summary())
    start_time_training = time.time()
    logging.info("Starting training")
    trainer = Trainer(
        train_set=training_set,
        test_set=evaluation_set,
        epochs=100,
        embedding_size=16,
        num_layer=24,
        save_file=model_save_path,
    )
    trainer.train()

    logging.info(f"Took {time_delta_now(start_time_training)} to train the model")

    logging.info(f"Saved model can be found at {model_save_path}")

    logging.info(f"Completed run after {time_delta_now(start_time)}.")


if __name__ == "__main__":
    run()
