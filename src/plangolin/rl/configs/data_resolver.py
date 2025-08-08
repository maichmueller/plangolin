import os
from argparse import ArgumentParser
from enum import auto

try:
    from enum import StrEnum  # Available in Python 3.11+
except ImportError:
    from strenum import StrEnum  # Backport for Python < 3.11
from pathlib import Path

from plangolin.rl.data_layout import InputData, OutputData

project_root_dir = Path(__file__).parent.parent.parent


class Parameter(StrEnum):
    pddl_domains_dir = auto()
    dataset_dir = auto()
    output_dir = auto()
    domain_name = auto()
    instances = auto()
    validation_instances = auto()


def from_parser_args(parser_args, exp_id: str):
    kwargs = {p.value: getattr(parser_args, p.value) for p in Parameter}
    output_dir = kwargs.pop(Parameter.output_dir.value)
    return InputData(**kwargs), OutputData(
        out_dir=output_dir,
        experiment_name=exp_id,
        domain_name=kwargs[Parameter.domain_name.value],
    )


def add_parser_args(
    parent_parser: ArgumentParser,
):
    parser = parent_parser.add_argument_group("Data Resolver")
    parser.add_argument(
        f"--{Parameter.pddl_domains_dir.value}",
        type=Path,
        required=False,
        help="Root directory of the input data.",
        default=os.path.join(project_root_dir, "data", "pddl_domains"),
    )
    parser.add_argument(
        f"--{Parameter.dataset_dir.value}",
        type=Path,
        required=False,
        help="Directory where to store/load datasets.",
        default=os.path.join(project_root_dir, "data", "flash_drives"),
    )
    parser.add_argument(
        f"--{Parameter.output_dir.value}",
        type=Path,
        required=False,
        help="Root directory of the output data.",
        default=os.path.join(project_root_dir, "out/"),
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
    parser.add_argument(
        f"--{Parameter.validation_instances}",
        type=str,
        required=False,
        help="List of instances to be used for validation."
        "If none are provided validation will be skipped",
    )
    return parent_parser
