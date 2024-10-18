import dataclasses
import datetime
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pymimir as mi


@dataclasses.dataclass
class OutputData:
    out_dir: Path
    experiment_name: str

    def __init__(
        self,
        out_dir: Path = Path("out"),
        experiment_name: str | None = None,
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
        self.out_dir.mkdir(exist_ok=True, parents=True)
        logging.info("Using " + str(self.out_dir) + " for output data.")
        self.experiment_name = experiment_name


class InputData:
    pddl_domains_dir: Path
    dataset_dir: Path
    domain: mi.Domain
    problems: List[mi.Problem]
    _instances: List[Path]
    validation_problems: Optional[List[mi.Problem]]
    _validation_instances: Optional[List[Path]]

    def __init__(
        self,
        domain_name: str,
        pddl_domains_dir: Path = Path("data/pddl_domains"),
        dataset_dir: Path = Path("data/flash_drives"),
        instances: Sequence[str] | None = None,
        validation_instances: List[str] | None = None,
        # if specified pddl_domains_dir and dataset_dir will be relative to root_dir
        root_dir: Optional[Path] = None,
    ):
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
            warnings.warn(
                "Domain input directory does not exist or is not a directory.\n"
                + str(self.pddl_domains_dir.absolute())
            )
        self._resolve_domain_and_instances(instances)
        if validation_instances is not None:
            self._resolve_validation_instances(validation_instances)
        else:
            self.validation_problems = None
            self._validation_instances = None

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

    def _resolve_instances(
        self, directory: Path, filter_list: List[str]
    ) -> Tuple[List[Path], List[mi.Problem]]:
        assert directory.is_dir()
        all_instances = list(directory.glob("*.pddl"))
        if len(all_instances) == 0:
            warnings.warn(
                "Could not find any *.pddl files in directory "
                + str(directory.absolute())
            )
            return [], []
        if filter_list is not None:
            filtered_instances = [
                instance
                for instance in all_instances
                if instance.stem in filter_list or instance.name in filter_list
            ]
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

    def _resolve_domain_and_instances(self, instances: List[str] | None = None):
        self.domain: mi.Domain = mi.DomainParser(str(self.domain_path)).parse()

        train_dir = self.pddl_domains_dir / "train"
        instances, problems = self._resolve_instances(train_dir, instances)
        self._instances = instances
        self.problems = problems

    def _resolve_validation_instances(
        self, eval_instances: List[str], use_train_dir: bool = True
    ):

        directory = "train" if use_train_dir else "eval"
        input_dir = self.pddl_domains_dir / directory
        instances, problems = self._resolve_instances(input_dir, eval_instances)
        self.validation_problems = problems
        self._validation_instances = instances
