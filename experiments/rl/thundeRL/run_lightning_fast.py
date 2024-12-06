#!/usr/bin/env python

import logging

import torch
import torch.nn

from rgnet.rl.thundeRL.cli_config import ThundeRLCLI


def cli_main():
    logging.getLogger().setLevel(logging.INFO)
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")
    cli = ThundeRLCLI()


if __name__ == "__main__":
    # https://discuss.pytorch.org/t/training-fails-due-to-memory-exhaustion-when-running-in-a-python-multiprocessing-process/202773/2
    torch.multiprocessing.set_start_method("fork", force=True)
    cli_main()
