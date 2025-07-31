#!/usr/bin/env python
import torch._dynamo

from rgnet.logging_setup import get_logger
from rgnet.utils.misc import increase_resource_limit

torch._dynamo.config.suppress_errors = True  # keep runtime errors from killing compile

# import torch.multiprocessing as mp
# spawn leads to deadlocks in some tests (e.g. test_iw.py and sometimes data_module.py)
# mp.set_start_method("spawn", force=True)

import argparse
import logging
import sys

from rgnet.rl.thundeRL import AtomValuesCLI, PolicyGradientCLI, ValueLearningCLI

CLI_REGISTRY = {
    "policy_gradient": PolicyGradientCLI,
    "atom_values": AtomValuesCLI,
    "supervised_value": ValueLearningCLI,
}


def cli_main():
    logging.getLogger().setLevel(logging.INFO)

    # --- Step 1: Pre-parse to extract --cli ---
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cli", help="Name of the CLI to run")
    known_args, remaining_args = pre_parser.parse_known_args()

    # # Remove the arguments from sys.argv so that the target CLI doesn't see it
    sys.argv = [sys.argv[0]]

    cli_name = known_args.cli.lower()
    if cli_name not in CLI_REGISTRY:
        print(f"Unknown CLI name: {cli_name}. Available: {list(CLI_REGISTRY.keys())}")
        exit(1)

    # Delay heavy imports until after CLI selection
    import torch

    CLI_CLASS = CLI_REGISTRY[cli_name]
    # torch.set_float32_matmul_precision("highest")
    # use the default file descriptor–based sharing to avoid mmap exhaustion
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    # torch.multiprocessing.set_sharing_strategy("file_system")
    get_logger(__name__).info(f"Running CLI: {CLI_CLASS.__name__}")
    cli = CLI_CLASS(args=remaining_args)  # Run the correct CLI


if __name__ == "__main__":
    # with torch.autograd.profiler.emit_nvtx():
    torch.multiprocessing.set_start_method("spawn", force=True)
    increase_resource_limit()
    cli_main()
