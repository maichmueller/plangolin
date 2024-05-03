import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pymimir as mi
import torch
from pymimir import Problem
from torch_geometric.data import Data

from rgnet.supervised import MultiInstanceSupervisedSet


def _lines_by_key(lines: List[str], key: str, is_list: bool = False) -> List[str]:
    """
    Extract lines between first line with BEGIN_{key} and first line END_{key}
    :param lines: List of lines (str) from where to extract the subset.
    :param key: One of OBJECTS, PREDICATES, GOAL, STATE, LABELED_STATE
    :param is_list: Whether to add the suffix _LIST to the key
    :return: The lines between the start and end token
    """
    begin_token = f"BEGIN_{key}" + ("_LIST" if is_list else "")
    end_token = f"END_{key}" + ("_LIST" if is_list else "")
    return lines[lines.index(begin_token) + 1 : lines.index(end_token)]


class DatasetParser:

    def __init__(self, domain: mi.Domain, problem: mi.Problem):
        self.domain = domain
        self.problem = problem
        self.predicate_ids = domain.get_predicate_id_map()
        self.obj_ids = {obj.id: obj for obj in problem.objects}

    def validate_predicates(self, lines: List[str]):
        for line in lines:
            pred_id, pred_str = line.strip().split()
            if (
                int(pred_id) not in self.predicate_ids
                or self.predicate_ids[int(pred_id)].name != pred_str
            ):
                raise ValueError(f"Predicate {pred_id} not found in domain")

    def validate_objects(self, lines: List[str]):
        # Assert that the ids and names of objects match between serialized and problem.
        for line in lines:
            obj_id, obj_str = line.split()
            obj_id = int(obj_id)
            if obj_id not in self.obj_ids or self.obj_ids[obj_id].name != obj_str:
                raise ValueError(
                    f"Object {obj_str} with id {obj_id} not found in problem "
                    f"with objects: {self.obj_ids}"
                )

    def parse_atoms(self, lines: List[str]) -> List[mi.Atom]:
        """
        Each line encodes one atom.
        1 2 3 -> encodes predicate with id 1 instantiated with objects with ids 2 and 3
        """
        atoms = []
        for line in lines:
            # We have to account for 0-arity predicates
            pred_id, *obj_ids = line.split() if line.count(" ") >= 1 else (line,)
            pred = self.predicate_ids[int(pred_id)]
            objs = [self.obj_ids[int(objId)] for objId in obj_ids]
            # This seems to be the only possibility to create atoms
            atom = pred.as_atom()
            for i, obj in enumerate(objs):
                atom = atom.replace_term(i, obj)
            atoms.append(atom)
        return atoms

    def parse_labeled_state(self, lines: List[str]) -> Tuple[int, mi.State]:
        """
        Example looks like:
        0 -> The label
        BEGIN_STATE -> List of atoms
        4 1 -> Encodes predicate with id 4 instantiated with object with id 1
        1 0
        2 0
        END_STATE
        :return: label, atoms
        """
        label = int(lines[0])
        atoms = self.parse_atoms(lines[2:-1])
        state = self.problem.create_state(atoms)
        return label, state

    def parse(self, txt_file: Path):
        lines = txt_file.read_text().splitlines()
        object_lines = _lines_by_key(lines, "OBJECTS")
        self.validate_objects(object_lines)
        pred_lines = _lines_by_key(lines, "PREDICATES")
        self.validate_predicates(pred_lines)

        # NOTE fact-list is ignored

        # Goals are never negated and are just a collection of atoms
        goal_lines = _lines_by_key(lines, "GOAL", is_list=True)
        goals = self.parse_atoms(goal_lines)

        state_lines = _lines_by_key(lines, "STATE", is_list=True)
        # Group every BEGIN_LABELED_STATE and END_LABELED_STATE together
        state_groups = []
        last_search_index = 0
        while True:
            try:
                next_start = state_lines.index("BEGIN_LABELED_STATE", last_search_index)
                next_end = state_lines.index("END_LABELED_STATE", next_start)
                state_groups.append(state_lines[next_start + 1 : next_end])
                last_search_index = next_end + 1
            except ValueError:
                break
        labeled_states = [self.parse_labeled_state(group) for group in state_groups]
        return goals, labeled_states


class SerializedDataset(MultiInstanceSupervisedSet):

    def __init__(
        self,
        domain: mi.Domain,
        problem_to_serialized_file: dict[mi.Problem, Path],
        **kwargs,
    ) -> None:
        """
        Instead of creating a dataset from a list of problems, we load the serialized files.
        :param domain: The domain to which the problems belong.
        :param problem_to_serialized_file: Mapping each problem to the corresponding serialized file.
        :param kwargs: Arguments passed to MultiInstanceSupervisedSet.
        """
        self.mapping = problem_to_serialized_file
        self.domain = domain
        super().__init__(**kwargs)

    def parse_problem(self, problem: Problem) -> List[Data]:
        file = self.mapping[problem]
        parser = DatasetParser(self.domain, problem)
        _, labeled_states = parser.parse(file)
        data_list = []
        for label, state in labeled_states:
            data: Data = self.encoder.to_pyg_data(self.encoder.encode(state))
            data.y = torch.tensor(label, dtype=torch.int64)
            data_list.append(data)
        return data_list


def match_problems(problems_dir: str, serialized_files_dir: str) -> dict[Path, Path]:
    """
    Assumes that every serializes path has the form {problem_name}_states.txt
    The problem_name is the exact file.stem of the problem pddl file.
    Serialized files without a corresponding problem are ignored.
    :param problems_dir: Directory containing pddl problems
    :param serialized_files_dir: Directory containing serialized files (*_states.txt)
    :return:  A mapping from problem-file to corresponding serialized-file
    """
    problem_files = Path(problems_dir).glob("*.pddl")
    problem_name_to_file: dict[str, Path] = {p.stem: p for p in problem_files}
    serialized_files = Path(serialized_files_dir).glob("*_states.txt")
    problem_to_serialized_file: dict[Path, Path] = {}
    for file in serialized_files:
        problem_name = file.stem[: -len("_states")]
        if problem_name not in problem_name_to_file:
            logging.warning(
                f"Serialized file {problem_name} not found in {problems_dir}"
            )
        problem_to_serialized_file[problem_name_to_file[problem_name]] = file
    if len(problem_to_serialized_file.keys()) != len(problem_name_to_file.keys()):
        mismatches = set(
            (p.stem for p in problem_to_serialized_file.keys())
        ).symmetric_difference(problem_name_to_file.keys())
        logging.warning(
            f"Mismatch between problems and serialized files: {mismatches} "
        )
    return problem_to_serialized_file
