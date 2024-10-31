from __future__ import annotations

import dataclasses
import datetime
import logging
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pymimir as mi

from experiments import ROOT_DIR


@dataclasses.dataclass
class OutputData:
    out_dir: Path
    experiment_name: str

    def __init__(
        self,
        out_dir: Path = Path("out"),
        experiment_name: str | None = None,
        ensure_new_out_dir: bool = False,
        root_dir: Path | None = None,
        domain_name: str | None = None,
    ):
        super().__init__()
        if root_dir:
            out_dir = root_dir / out_dir
        elif out_dir == Path("out"):
            warnings.warn(
                "If the root directory is not specified an absolut path for"
                " 'directory' should be specified."
            )
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")

        if domain_name is not None:
            self.out_dir = out_dir / domain_name / experiment_name
        else:
            self.out_dir = out_dir / experiment_name

        if ensure_new_out_dir:
            suffix = 0
            new_out_dir = self.out_dir
            while new_out_dir.exists():
                new_out_dir = Path(str(self.out_dir.absolute()) + "_" + str(suffix))
                suffix += 1
            self.out_dir = new_out_dir

        self.out_dir.mkdir(exist_ok=True, parents=True)
        logging.info("Using " + str(self.out_dir) + " for output data.")
        self.experiment_name = experiment_name


def _all_or_filter(instances: List[str] | Literal["all"]):
    return instances if isinstance(instances, list) else None


class InputData:
    pddl_domains_dir: Path
    dataset_dir: Path
    domain: mi.Domain
    problems: List[mi.Problem]
    _instances: List[Path]
    validation_problems: Optional[List[mi.Problem]]
    _validation_instances: Optional[List[Path]]
    test_problems: Optional[List[mi.Problem]]
    _test_instances: Optional[List[Path]]

    def __init__(
        self,
        domain_name: str,
        pddl_domains_dir: Path = Path("data/pddl_domains"),
        dataset_dir: Path = Path("data/flash_drives"),
        train_subdir: str = "train",
        eval_subdir: Optional[str] = None,
        test_subdir: Optional[str] = None,
        instances: List[str] | Literal["all"] = "all",
        validation_instances: Optional[List[str]] | Literal["all"] = None,
        test_instances: Optional[List[str]] | Literal["all"] = None,
        # if specified pddl_domains_dir and dataset_dir will be relative to root_dir
        root_dir: Optional[Path] = ROOT_DIR,
    ):
        """
        Manages the data layout for input data to RL experiments.
        The directories can either be specified as absolute paths or relative to the root_dir.
        In basic layout inside pddl_domains_dir looks like
        domain_name
            domain.pddl
            train
                instance1.pddl
        the domain is expected to be at pddl_domains_dir / domain_name /  "domain.pddl".
        Evaluation and test problems can be either in a separate directories or in the train directory too.
        Pass a list of specific instance names, or use "all" to include all available instances for each parameter.
        Specifying validation or test instances is optional.

        :param domain_name: The name of the domain, used to construct the full pddl_domains directory.
        :param pddl_domains_dir: path to the pddl_resources (default: rgnet/data/pddl_domains)
        :param dataset_dir: where to store datasets (default: rgnet/data/flash_drives)
        :param train_subdir: the name of the subdirectory where the problem pddl files are located. (default: "train")
        :param eval_subdir: the subdirectory where the validation problem files are located. (default: "train")
        :param test_subdir: the subdirectory where the testing problem files are located. (default: "train")
        :param instances: Either an explicit list of file names (or stems) or "all". (default: "all")
        :param validation_instances: None to skip validation, an explicit list or "all". (default: None)
        :param test_instances: None to skip testing, an explicit list of instances or "all". (default: None)
        :param root_dir: Optional parent directory for both pddl_domains_dir and dataset_dir.
            If specified the parameter pddl_domains_dir and dataset_dir will be interpreted as relative to root_dir.
            (default: the project source directory)
        """
        self.train_subdir = train_subdir
        self.eval_subdir = eval_subdir or self.train_subdir
        self.test_subdir = test_subdir or self.train_subdir
        if root_dir:
            pddl_domains_dir = root_dir / pddl_domains_dir
            dataset_dir = root_dir / dataset_dir
        else:
            if pddl_domains_dir.parts[0] == "data" or dataset_dir.parts[0] == "data":
                warnings.warn(
                    "If the root directory is not specified absolute paths for"
                    " 'pddl_domains_dir' and 'dataset_dir' should be specified."
                )
        self.dataset_dir = dataset_dir
        if not self.dataset_dir.exists():
            logging.info(
                "Creating missing dataset dir at " + str(self.dataset_dir.absolute())
            )
            self.dataset_dir.mkdir(parents=True)
        self.pddl_domains_dir = pddl_domains_dir / domain_name
        if not self.pddl_domains_dir.exists() or not self.pddl_domains_dir.is_dir():
            raise ValueError(
                "Domain input directory does not exist or is not a directory.\n"
                + str(self.pddl_domains_dir.absolute())
            )
        self._resolve_domain_and_instances(instances)
        self._resolve_validation_instances(validation_instances)
        self._resolve_test_instances(test_instances)

        # We load state space lazily as it might be quite expensive
        self._spaces: Optional[List[mi.StateSpace]] = None
        self._validation_spaces: Optional[List[mi.StateSpace]] = None
        self.space_by_problem: dict[mi.Problem, mi.StateSpace] = dict()

    def _get_or_load_space(self, problem: mi.Problem) -> mi.StateSpace:
        if problem not in self.space_by_problem:
            self.space_by_problem[problem] = mi.StateSpace.new(
                problem, mi.GroundedSuccessorGenerator(problem)
            )
        return self.space_by_problem.get(problem, None)

    @property
    def spaces(self):
        if self._spaces is None:
            self._spaces = [
                self._get_or_load_space(problem) for problem in self.problems
            ]
        return self._spaces

    @property
    def validation_spaces(self):
        if self.validation_problems is None:
            raise ValueError(
                "Tried to access validation spaces but no validation problems were provided"
            )
        if self._validation_spaces is None:
            self._validation_spaces = [
                self._get_or_load_space(problem) for problem in self.validation_problems
            ]
        return self._validation_spaces

    @property
    def domain_path(self) -> Path:
        return self.pddl_domains_dir / "domain.pddl"

    @property
    def problem_paths(self) -> List[Path]:
        return self._instances

    @property
    def validation_problem_paths(self) -> Optional[List[Path]]:
        return self._validation_instances

    @property
    def test_problem_paths(self) -> Optional[List[Path]]:
        return self._test_instances

    def _resolve_instances(
        self, directory: Path, filter_list: List[str] | None
    ) -> Tuple[List[Path], List[mi.Problem]]:
        """
        Collects pddl problems from a directory, potentially filters them, and parses them as pymimir Problems.
        :param directory: The directory in which the .pddl problems are in. Note the domain.pddl should not be in this directory.
        :param filter_list: The list of instances to filter for, can be the full file name or only the stem. If None then all files will be used.
        :return: The collected pddl files as paths and the respective parsed problems.
        """
        assert directory.is_dir(), f"Given directory {directory} is not a directory."
        all_instances = list(directory.glob("*.pddl"))
        if any(instance_path.name == "domain.pddl" for instance_path in all_instances):
            warnings.warn(
                f"Found domain.pddl in {directory.resolve()} this will likely result in a parse exception. Only problems should be placed in this directory."
            )
        if len(all_instances) == 0:
            warnings.warn(
                "Could not find any *.pddl files in directory "
                + str(directory.absolute())
            )
            return [], []
        if filter_list is not None:
            # Ensure that the resolved paths are in the same order as filter_list
            filtered_instances = []
            for file_name in filter_list:
                if not file_name.endswith(".pddl"):
                    file_name += ".pddl"
                instance_path = directory / file_name
                if not instance_path.exists():
                    warnings.warn(
                        f"Could not find {(directory / file_name).absolute()}"
                    )
                else:
                    filtered_instances.append(instance_path)
            if len(filtered_instances) == 0:
                warnings.warn(
                    "Filter matched no instances."
                    f"Tried to filter for: {filter_list} but found {[i.name for i in all_instances]}"
                )
                return [], []
            all_instances = filtered_instances

        return all_instances, [
            mi.ProblemParser(str(instance)).parse(self.domain)
            for instance in all_instances
        ]

    def _resolve_domain_and_instances(self, instances: List[str] | Literal["all"]):
        self.domain: mi.Domain = mi.DomainParser(str(self.domain_path)).parse()

        train_dir = self.pddl_domains_dir / self.train_subdir
        found_instances, problems = self._resolve_instances(
            train_dir, _all_or_filter(instances)
        )
        self._instances = found_instances
        self.problems = problems

    def _resolve_validation_instances(
        self,
        validation_instances: Optional[List[str]] | Literal["all"],
    ):
        if validation_instances is None:
            self.validation_problems = None
            self._validation_instances = None
            return
        else:
            input_dir = self.pddl_domains_dir / self.eval_subdir
            instances, problems = self._resolve_instances(
                input_dir, _all_or_filter(validation_instances)
            )
            self.validation_problems = problems
            self._validation_instances = instances

    def _resolve_test_instances(
        self,
        test_instances: Optional[List[str]] | Literal["all"],
    ):
        if test_instances is None:
            self.test_problems = None
            self._test_instances = None
            return
        else:
            input_dir = self.pddl_domains_dir / self.test_subdir
            instances, problems = self._resolve_instances(
                input_dir, _all_or_filter(test_instances)
            )
            self.test_problems = problems
            self._test_instances = instances
