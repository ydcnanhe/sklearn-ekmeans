"""Public API for the :mod:`sklekmeans` package.

The package exposes clustering estimators implementing the Equilibrium K-Means
objective and its semi-supervised extension:

* :class:`~sklekmeans.EKMeans` – batch Equilibrium K-Means.
* :class:`~sklekmeans.MiniBatchEKMeans` – mini-batch variant suitable for
    larger datasets.
* :class:`~sklekmeans.SSEKM` – semi-supervised EKMeans (batch) with guidance
    matrix ``F`` and mixing factor ``theta``.
* :class:`~sklekmeans.MiniBatchSSEKM` – mini-batch semi-supervised variant.

All estimators follow the scikit-learn estimator API (``fit``, ``predict``,
``transform``) and provide additional helpers (``membership`` and
``fit_membership``) returning soft assignment matrices.
"""

from ._ekmeans import EKMeans, MiniBatchEKMeans
from ._ssekmeans import SSEKM, MiniBatchSSEKM

__all__ = ["EKMeans", "MiniBatchEKMeans", "SSEKM", "MiniBatchSSEKM"]

# Light-weight version attribute for now; adjust if setuptools_scm is adopted.
__version__ = "0.2.0"
