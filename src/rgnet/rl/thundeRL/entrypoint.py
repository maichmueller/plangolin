#!/usr/bin/env python

import argparse
import logging
import sys

from rgnet.rl.thundeRL import AtomValuesCLI, PolicyGradientCLI

# --- Step 1: Pre-parse to extract --cli ---
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument(
    "--cli", default="policy_gradient", help="Name of the CLI to run"
)
known_args, remaining_args = pre_parser.parse_known_args()

# # Remove the --cli argument from sys.argv so that the target CLI doesn't see it
sys.argv = [sys.argv[0]] + remaining_args

# Optional: import based on CLI choice
# Example: map --cli name to actual CLI class
CLI_REGISTRY = {
    "policy_gradient": PolicyGradientCLI,
    "atom_values": AtomValuesCLI,
}

cli_name = known_args.cli.lower()
if cli_name not in CLI_REGISTRY:
    print(f"Unknown CLI name: {cli_name}. Available: {list(CLI_REGISTRY.keys())}")
    sys.exit(1)

# Delay heavy imports until after CLI selection
import torch

CLI_CLASS = CLI_REGISTRY[cli_name]


def increase_resource_limit():
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 3e4:
        new_soft = int(3e4)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logging.info(
            f"Changing resource limits to: [{soft = } --> {new_soft = }, {hard = }]"
        )
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logging.info(f"Resource limits: [{soft = }, {hard = }]")


def cli_main():
    logging.getLogger().setLevel(logging.INFO)
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")
    logging.info(f"Running CLI: {CLI_CLASS.__name__}")
    cli = CLI_CLASS()  # Run the correct CLI


if __name__ == "__main__":
    increase_resource_limit()
    cli_main()
