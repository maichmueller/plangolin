#!/usr/bin/env python

import logging

import torch
import torch.nn

from .cli_config import ThundeRLCLI


def increase_resource_limit():
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 3e4:
        new_soft = int(3e4)  # arbitrary, increase if necessary
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
    cli = ThundeRLCLI()


if __name__ == "__main__":
    increase_resource_limit()
    cli_main()
