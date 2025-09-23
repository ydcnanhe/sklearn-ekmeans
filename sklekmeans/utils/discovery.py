"""Discovery utilities for sklekmeans package."""

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils._testing import ignore_warnings

_MODULE_TO_IGNORE = {"tests"}


def all_estimators(type_filter=None):
    """Get a list of all estimators from `sklekmeans`.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "cluster", "transformer"} or list, default=None
        Filter the estimators by their mixin type.
    """

    def is_abstract(c):
        if not hasattr(c, "__abstractmethods__"):
            return False
        return bool(c.__abstractmethods__)

    all_classes = []
    root = str(Path(__file__).parent.parent)  # sklekmeans package
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklekmeans."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [(name, est_cls) for name, est_cls in classes if not name.startswith("_")]
            all_classes.extend(classes)

    all_classes = set(all_classes)
    estimators = [c for c in all_classes if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")]
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', 'regressor', 'transformer', 'cluster' or None, got " f"{repr(type_filter)}.")
    return sorted(set(estimators), key=itemgetter(0))


def all_displays():
    """Get a list of all displays from `sklekmeans`."""
    all_classes = []
    root = str(Path(__file__).parent.parent)  # sklekmeans package
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklekmeans."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, display_class)
                for name, display_class in classes
                if (not name.startswith("_") and name.endswith("Display"))
            ]
            all_classes.extend(classes)
    return sorted(set(all_classes), key=itemgetter(0))


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False
    if item.__name__.startswith("_"):
        return False
    mod = item.__module__
    if not mod.startswith("sklekmeans.") or mod.endswith("estimator_checks"):
        return False
    return True


def all_functions():
    """Get a list of all functions from `sklekmeans`."""
    all_functions_list = []
    root = str(Path(__file__).parent.parent)  # sklekmeans package
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklekmeans."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            module = import_module(module_name)
            functions = inspect.getmembers(module, _is_checked_function)
            functions = [
                (func.__name__, func)
                for name, func in functions
                if not name.startswith("_")
            ]
            all_functions_list.extend(functions)
    return sorted(set(all_functions_list), key=itemgetter(0))

__all__ = ["all_estimators", "all_displays", "all_functions"]
