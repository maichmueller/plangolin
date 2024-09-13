from argparse import ArgumentParser
from enum import StrEnum, auto
from pathlib import Path

from experiments.rl.data_resolver import DataResolver


class Parameter(StrEnum):
    input_dir = auto()
    output_dir = auto()
    domain_name = auto()
    instances = auto()


def from_parser_args(parser_args, exp_id: str):
    kwargs = {p.value: getattr(parser_args, p.value) for p in Parameter}
    return DataResolver(**kwargs, exp_id=exp_id)


def add_parser_args(
    parent_parser: ArgumentParser,
):
    parser = parent_parser.add_argument_group("Data Resolver")
    parser.add_argument(
        f"--{Parameter.input_dir.value}",
        type=Path,
        required=False,
        help="Root directory of the input data.",
        default="../../data/pddl_domains",
    )
    parser.add_argument(
        f"--{Parameter.output_dir.value}",
        type=Path,
        required=False,
        help="Root directory of the output data.",
        default="../../out/",
    )
    parser.add_argument(
        f"--{Parameter.domain_name.value}",
        type=str,
        required=True,
        help="Name of the domain.",
    )
    parser.add_argument(
        f"--{Parameter.instances.value}",
        type=str,
        nargs="*",
        help="List of instances to be used."
        "If none are provided all instances found in the directory will be used.",
    )
    return parent_parser
