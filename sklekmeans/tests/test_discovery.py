"""Tests for discovery utilities in sklekmeans."""
from sklekmeans.utils import all_displays, all_estimators, all_functions


def test_all_estimators():
    ests = all_estimators()
    # We expect 3 template estimators
    assert len(ests) == 3


def test_all_displays():
    # No displays defined yet
    assert len(all_displays()) == 0


def test_all_functions():
    funcs = all_functions()
    # No standalone functions currently implemented
    assert len(funcs) == 0
