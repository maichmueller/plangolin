#!/usr/bin/env python

import torch._dynamo
from jsonargparse._util import import_object

from rgnet.logging_setup import get_logger
from rgnet.utils.system import increase_resource_limit

torch._dynamo.config.suppress_errors = True  # keep runtime errors from killing compile


import argparse
import logging

logger = get_logger(__name__)


def cli_main():
    logging.getLogger().setLevel(logging.INFO)
    # use the default file descriptor–based sharing to avoid mmap exhaustion
    # torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    # torch.set_float32_matmul_precision("highest")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cli", help="Name of the CLI to run")
    known_args, remaining_args = pre_parser.parse_known_args()

    cli_name = known_args.cli
    logger.info(f"Attempting to let jsonargparse load CLI from dot-path: {cli_name}")
    cli_class = import_object(cli_name)
    logger.info(f"Running CLI: {cli_class!r}")
    cli = cli_class(args=remaining_args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    increase_resource_limit()
    # with torch.autograd.profiler.emit_nvtx():
    cli_main()
