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
