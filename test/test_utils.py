from rgnet.utils import ftime


def test_ftime():
    assert "100us" == ftime(0.0001)
    assert "100ms" == ftime(0.1)
    assert "14s" == ftime(14)
    assert "01:00m" == ftime(60)
    assert "01:01m" == ftime(61)
    assert "1:00:00h" == ftime(3600)
    assert "1:00:01h" == ftime(3601)
    assert "01:00m" == ftime(60.1)


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
