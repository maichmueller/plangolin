from __future__ import annotations

import datetime
import itertools
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import xmimir as xmi
from rgnet.utils.plan import Plan, parse_fd_plan
from xmimir import XDomain, XProblem, XStateSpace


@dataclass
class OutputData:
    out_dir: Path
    experiment_name: str

    def __init__(
        self,
        out_dir: Path | str = Path("out"),
        experiment_name: str | None = None,
        ensure_new_out_dir: bool = False,
        root_dir: Path | None = None,
        domain_name: str | None = None,
        output_dir_order: Literal["domain"] | Literal["experiment"] = "domain",
    ):
        super().__init__()
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            if root_dir is None:
                logging.info(
                    f"The root directory is not specified and a relative path is given for '{out_dir=}'. "
                    f"Defaulting to relative the current working directory: '{os.getcwd() / out_dir}'."
                )
            else:
                out_dir = root_dir / out_dir

        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")

        if domain_name is not None:
            if output_dir_order == "domain":
                self.out_dir = out_dir / domain_name / experiment_name
            elif output_dir_order == "experiment":
                self.out_dir = out_dir / experiment_name / domain_name
            else:
                raise ValueError(
                    f"Invalid 'output_dir_order'. Must be 'domain' or 'experiment'. Given {output_dir_order=}"
                )
        else:
            self.out_dir = out_dir / experiment_name

        # Create the output directly, there could be multiple experiments with
        # the same experiment name running in parallel.
        try:
            # If ensure_new_out_dir it is not okay that the directory exists.
            self.out_dir.mkdir(exist_ok=not ensure_new_out_dir, parents=True)
        except FileExistsError:
            suffix = 0
            while True:
                try:
                    unique_dir = self.out_dir.parent / f"{experiment_name}_{suffix}"
                    unique_dir.mkdir(parents=True, exist_ok=False)
                    self.out_dir = unique_dir
                    experiment_name = f"{experiment_name}_{suffix}"
                    break
                except FileExistsError:
                    suffix += 1

        logging.info("Using " + str(self.out_dir) + " for output data.")
        self.experiment_name = experiment_name


def _all_or_filter(instances: List[str] | Literal["all"]):
    return instances if isinstance(instances, list) else None


class InputData:
    pddl_domains_dir: Path
    dataset_dir: Path
    domain: XDomain
    problems: List[XProblem]
    _instances: List[Path]
    validation_problems: Optional[List[XProblem]]
    _validation_instances: Optional[List[Path]]
    test_problems: Optional[List[XProblem]]
    _test_instances: Optional[List[Path]]
    # Not guaranteed to have a plan for every problem.
    # Shared across training, validation and test problems.
    plan_by_problem: Dict[XProblem, Plan]

    def __init__(
        self,
        domain_name: str,
        pddl_domains_dir: Path = Path("data/pddl_domains"),
        dataset_dir: Path = Path("data/flash_drives"),
        train_subdir: str = "train",
        eval_subdir: Optional[str] = None,
        test_subdir: Optional[str] = None,
        plan_subdir: Optional[str] = None,
        instances: List[str] | Literal["all"] = "all",
        validation_instances: Optional[List[str]] | Literal["all"] = None,
        test_instances: Optional[List[str]] | Literal["all"] = None,
        # if specified pddl_domains_dir and dataset_dir will be relative to root_dir
        root_dir: Optional[Path] = None,
    ):
        """
        Manages the data layout for input data to RL experiments.
        The directories can either be specified as absolute paths or relative to the root_dir.
        In basic layout inside `pddl_domains_dir` looks like
        domain_name
            domain.pddl
            train
                instance1.pddl
        the domain is expected to be at pddl_domains_dir / domain_name /  "domain.pddl".
        Evaluation and test problems can be either in a separate directories or in the train directory too.
        Pass a list of specific instance names, or use "all" to include all available instances for each parameter.
        Specifying validation or test instances is optional.

        :param domain_name: The name of the domain, used to construct the full pddl_domains directory.
        :param pddl_domains_dir: path to the pddl_resources (default: project/data/pddl_domains)
        :param dataset_dir: where to store datasets (default: project/data/flash_drives)
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
        self.plan_subdir = plan_subdir or self.test_subdir
        self.plan_subdir_was_specified = plan_subdir is not None
        if root_dir:
            pddl_domains_dir = root_dir / pddl_domains_dir
            dataset_dir = root_dir / dataset_dir
        else:
            if pddl_domains_dir.parts[0] == "data" or dataset_dir.parts[0] == "data":
                warnings.warn(
                    "If the root directory is not specified absolute paths for"
                    " 'pddl_domains_dir' and 'dataset_dir' should be specified."
                )
        self.dataset_dir = dataset_dir / domain_name
        if not self.dataset_dir.parent.exists():
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
        self.plan_by_problem = self._look_for_plans()

        # We load state space lazily as it might be quite expensive
        self._spaces: Optional[List[XStateSpace]] = None
        self._validation_spaces: Optional[List[XStateSpace]] = None
        self._space_by_problem: dict[XProblem, XStateSpace] = dict()

    def get_or_load_space(
        self, problem: XProblem, max_expanded: int | None = None
    ) -> XStateSpace:
        if problem not in self._space_by_problem:
            self._space_by_problem[problem] = XStateSpace(
                self.domain.filepath,
                problem.filepath,
                max_num_states=max_expanded or 1_000_000,
            )
        return self._space_by_problem.get(problem, None)

    @property
    def spaces(self):
        if self._spaces is None:
            self._spaces = [
                self.get_or_load_space(problem) for problem in self.problems
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
                self.get_or_load_space(problem) for problem in self.validation_problems
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
    ) -> Tuple[List[Path], List[XProblem]]:
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
            logging.warning(
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
                    logging.warning(
                        f"Could not find {(directory / file_name).absolute()}"
                    )
                else:
                    filtered_instances.append(instance_path)
            if len(filtered_instances) == 0:
                logging.warning(
                    "Filter matched no instances."
                    f"Tried to filter for: {filter_list} but found {[i.name for i in all_instances]}"
                )
                return [], []
            all_instances = filtered_instances

        problems = []
        for instance in all_instances:
            try:
                problems.append(xmi.parse(str(self.domain_path), str(instance))[1])
            except RuntimeError as e:
                domain_warn_message = (
                    "Problem file is likely a domain."
                    "Please make sure that domain files are outside of the problem subdirectories."
                    "The expected layout can be found in the documentation of InputData."
                )
                warn_message = f"Could not parse problem {instance}.{repr(e)}"
                if instance.stem.lower() == "domain":
                    logging.warning(f"{domain_warn_message}\n{warn_message}")
                else:
                    logging.warning(warn_message)

        return all_instances, problems

    def _resolve_domain_and_instances(self, instances: List[str] | Literal["all"]):
        train_dir = self.pddl_domains_dir / self.train_subdir
        found_instances, problems = self._resolve_instances(
            train_dir, _all_or_filter(instances)
        )
        self.domain: xmi.XDomain = problems[0].domain
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

    def _look_for_plans(self) -> Dict[XProblem, Plan]:
        plan_dir = self.pddl_domains_dir / self.plan_subdir
        plan_files = list(plan_dir.glob("*.plan"))
        if len(plan_files) == 0 and self.plan_subdir_was_specified:
            warnings.warn(
                f"Could not find .plan files in {plan_dir} even though subdirectory was explicitly set to {self.plan_subdir}"
            )
            return dict()

        all_problems = set(
            itertools.chain.from_iterable(
                problems
                for problems in [
                    self.problems,
                    self.validation_problems,
                    self.test_problems,
                ]
                if problems
            )
        )
        problem_by_stem: Dict[str, XProblem] = {
            Path(p.filepath).stem: p for p in all_problems
        }
        problem_to_plan: Dict[XProblem, Plan] = {}
        for plan_file in plan_files:
            pddl_file = plan_file.stem.removesuffix(".pddl")
            problem = problem_by_stem.get(pddl_file)
            if problem is None:
                logging.debug(
                    f"Could not match plan file to problem. Plan name {pddl_file}.\n"
                    f"\tCandidates: {problem_by_stem.keys()}"
                )
                continue
            problem_to_plan[problem] = parse_fastdownward_plan(plan_file, problem)
        return problem_to_plan
