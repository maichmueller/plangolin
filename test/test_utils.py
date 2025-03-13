import pathlib

from rgnet.utils import ftime, import_all_from
from xmimir import XDomain, XProblem


def test_ftime():
    assert "100us" == ftime(0.0001)
    assert "100ms" == ftime(0.1)
    assert "14s" == ftime(14)
    assert "01:00m" == ftime(60)
    assert "01:01m" == ftime(61)
    assert "1:00:00h" == ftime(3600)
    assert "1:00:01h" == ftime(3601)
    assert "01:00m" == ftime(60.1)


def test_import_all_from():
    path = "test/pddl_instances/blocks"
    domain, problems = import_all_from(
        "test/pddl_instances/blocks", domain_filename="domain"
    )
    assert isinstance(domain, XDomain)
    assert all(
        isinstance(prob, XProblem) and prob.domain.filepath == domain.filepath
        for prob in problems
    )
    # -1 as one pddl file is the domain
    assert len(problems) == len(list(pathlib.Path(path).glob("*.pddl"))) - 1
