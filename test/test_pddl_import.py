import pathlib

from pymimir import Domain, Problem

from rgnet.utils import import_all_from


def test_import_all_from():
    path = "test/pddl_instances/blocks"
    domain, problems = import_all_from(
        "test/pddl_instances/blocks", domain_name="domain"
    )
    assert isinstance(domain, Domain)
    assert all(isinstance(p, Problem) and p.domain == domain for p in problems)
    # -1 as one pddl file is the domain
    assert len(problems) == len(list(pathlib.Path(path).glob("*.pddl"))) - 1
