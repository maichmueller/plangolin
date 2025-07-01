import os
from pathlib import Path
from test.fixtures import small_blocks  # noqa: F401, F403

import pytest

from xmimir import *  # noqa: F401, F403
from xmimir.wrappers import atom_str_template


def test_atom_repr_matches_pymimir(small_blocks):
    space = small_blocks[0]
    for state in space:
        for atom in state.atoms():
            assert atom_str_template.render(
                predicate=atom.predicate.name, objects=atom.objects
            ) == str(atom.base)


@pytest.mark.parametrize(("domain", "problem"), [("blocks", "medium")])
def test_build_space_with_options(domain, problem):
    source_dir = Path("" if os.getcwd().endswith("/test") else "test/")
    domain_path = source_dir / "pddl_instances" / domain / "domain.pddl"
    problem_path = source_dir / "pddl_instances" / domain / f"{problem}.pddl"
    with pytest.raises(ValueError):
        XStateSpace(
            domain_path, problem_path, max_num_states=10, use_unit_cost_one=False
        )
