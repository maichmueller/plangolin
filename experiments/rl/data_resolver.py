import warnings
from pathlib import Path
from typing import List, Optional

import pymimir as mi


class DataResolver:
    input_dir: Path
    output_dir: Path
    exp_id: str
    domain: mi.Domain
    problems: List[mi.Problem]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        exp_id: str,
        domain_name: str,
        instances: List[str] | None = None,
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
        output_dir_root = output_dir / domain_name
        output_dir_root.mkdir(parents=False, exist_ok=True)
        self._resolve_out_dir(output_dir_root, exp_id)

        # We load state space lazily as it might be quite expensive
        self._spaces: Optional[List[mi.StateSpace]] = None

    @property
    def spaces(self):
        if self._spaces is None:
            self._spaces = [
                mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
                for problem in self.problems
            ]
        return self._spaces

    def _resolve_domain_and_instances(self, instances_list: List[str] | None = None):
        self.domain: mi.Domain = mi.DomainParser(
            str(self.input_dir / "domain.pddl")
        ).parse()

        train_dir = self.input_dir / "train"
        all_instances = list(train_dir.glob("*.pddl"))
        if instances_list is not None:
            filtered_instances = [
                instance
                for instance in all_instances
                if instance.stem in instances_list or instance.name in instances_list
            ]
            instances = filtered_instances
        else:
            instances = all_instances

        self.problems: List[mi.Problem] = [
            mi.ProblemParser(str(instance)).parse(self.domain) for instance in instances
        ]
        if len(self.problems) == 0:
            warnings.warn(
                "No instances found in the directory.\n"
                + (
                    f"Using filter {instances_list}"
                    if instances_list is not None
                    else ""
                )
                + f"Found {list(i.name for i in all_instances)} before filtering."
            )

    def _resolve_out_dir(self, out_dir_root: Path, exp_id: str):
        count = 0
        self.output_dir = out_dir_root / exp_id
        while self.output_dir.exists():
            count += 1
            self.output_dir = out_dir_root / f"{exp_id}_{count}"
        self.output_dir.mkdir(parents=False, exist_ok=False)
