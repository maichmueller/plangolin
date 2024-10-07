import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pymimir as mi


class DataResolver:
    input_dir: Path
    output_dir: Path
    exp_id: str
    domain: mi.Domain
    problems: List[mi.Problem]
    _instances: List[Path]
    validation_problems: Optional[List[mi.Problem]]
    _validation_instances: Optional[List[Path]]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        exp_id: str,
        domain_name: str,
        instances: List[str] | None = None,
        validation_instances: List[str] | None = None,
    ):
        self.exp_id: str = exp_id
        if not any(p.name == "rgnet" for p in input_dir.parent.parents):
            warnings.warn(
                "Input directory is not a sub-directory of rgnet.\n"
                + str(input_dir.absolute())
            )
        self.input_dir = input_dir / domain_name
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            warnings.warn(
                "Domain input directory does not exist or is not a directory.\n"
                + str(self.input_dir.absolute())
            )
        self._resolve_domain_and_instances(instances)
        if validation_instances is not None:
            self._resolve_validation_instances(validation_instances)
        else:
            self.validation_problems = None
            self._validation_instances = None

        output_dir_root = output_dir / domain_name
        output_dir_root.mkdir(parents=False, exist_ok=True)
        self._resolve_out_dir(output_dir_root, exp_id)

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
        return self.input_dir / "domain.pddl"

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

        train_dir = self.input_dir / "train"
        instances, problems = self._resolve_instances(train_dir, instances)
        self._instances = instances
        self.problems = problems

    def _resolve_validation_instances(
        self, eval_instances: List[str], use_train_dir: bool = True
    ):

        directory = "train" if use_train_dir else "eval"
        input_dir = self.input_dir / directory
        instances, problems = self._resolve_instances(input_dir, eval_instances)
        self.validation_problems = problems
        self._validation_instances = instances

    def _resolve_out_dir(self, out_dir_root: Path, exp_id: str):
        count = 0
        self.output_dir = out_dir_root / exp_id
        while self.output_dir.exists():
            count += 1
            self.output_dir = out_dir_root / f"{exp_id}_{count}"
        self.output_dir.mkdir(parents=False, exist_ok=False)
