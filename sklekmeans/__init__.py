"""Public API for the :mod:`sklekmeans` package.

The package currently exposes two clustering estimators implementing the
Equilibrium K-Means objective:

* :class:`~sklekmeans.EKMeans` – batch Equilibrium K-Means.
* :class:`~sklekmeans.MiniBatchEKMeans` – mini-batch variant suitable for
    larger datasets.

Both estimators follow the scikit-learn estimator API (``fit``, ``predict``,
``transform``) and provide additional helpers (``membership`` and
``fit_membership``) returning soft assignment matrices.
"""

from ._ekmeans import EKMeans, MiniBatchEKMeans

__all__ = ["EKMeans", "MiniBatchEKMeans"]

# Light-weight version attribute for now; adjust if setuptools_scm is adopted.
__version__ = "0.1.0"
