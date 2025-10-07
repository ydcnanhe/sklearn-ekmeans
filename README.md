sklekmeans - Equilibrium K-Means for scikit-learn
=================================================

[![Unit Tests](https://github.com/ydcnanhe/sklearn-ekmeans/actions/workflows/python-app.yml/badge.svg)](https://github.com/ydcnanhe/sklearn-ekmeans/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/ydcnanhe/sklearn-ekmeans/graph/badge.svg)](https://codecov.io/gh/ydcnanhe/sklearn-ekmeans)
[![docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://ydcnanhe.github.io/sklearn-ekmeans)
[![PyPI version](https://img.shields.io/pypi/v/sklekmeans.svg)](https://pypi.org/project/sklekmeans/)
[![Python versions](https://img.shields.io/pypi/pyversions/sklekmeans.svg)](https://pypi.org/project/sklekmeans/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

`sklekmeans` provides batch and mini-batch implementations of the
Equilibrium K-Means (EKMeans) clustering algorithm. The method introduces
an equilibrium weighting scheme that can yield improved robustness on
imbalanced datasets compared to standard k-means. The API is compatible
with sklearn estimators.

Features
--------
* Drop-in scikit-learn compatible estimators: `EKMeans`, `MiniBatchEKMeans`, `SSEKM`, `MiniBatchSSEKM` (semi-supervised).
* Supports Euclidean and Manhattan distances.
* Heuristic alpha selection via `alpha='dvariance'` (default).
* Mini-batch variant with accumulation or online update modes.
* Soft memberships (`membership`) and equilibrium weights (`W_`).
* Semi-supervised learning via a prior matrix (`prior_matrix`, shape `(n_samples, n_clusters)`), with supervision strength `theta` (default `theta='auto' = |N|/|S|`).

Installation
------------
The package is available on PyPI. Install the base package:

```bash
pip install sklekmeans
```

Optional extras:

- With numba acceleration (recommended for speed):

```bash
pip install "sklekmeans[speed]"
```

From source (latest main):

- Basic installation

```bash
git clone https://github.com/ydcnanhe/sklearn-ekmeans.git
cd sklearn-ekmeans
pip install .
```

- Or in editable mode

```bash
pip install -e .
```

- With numba acceleration

```bash
pip install -e .[speed]
```

- Development tools (tests, lint):

```bash
pip install -e .[dev]
```

- Docs build dependencies:

```bash
pip install -e .[docs]
```

- Everything (dev + docs + speed):

```bash
pip install -e .[all]
```

Quick Start
-----------
```python
from sklekmeans import EKMeans
import numpy as np

X = np.random.rand(200, 2)
ekm = EKMeans(n_clusters=3, random_state=0).fit(X)
print(ekm.cluster_centers_)
```

Mini-batch variant with multiple initializations and selection of the best run:

```python
from sklekmeans import MiniBatchEKMeans
mb = MiniBatchEKMeans(n_clusters=3, batch_size=256, max_epochs=20, n_init=5, random_state=0)
mb.fit(X)
print(mb.cluster_centers_)
```

Semi-supervised variant (SSEKM)
-------------------------------
Use `prior_matrix` to inject partial labels or weak supervision. Unlabeled rows are all zeros; labeled rows provide per-class probabilities (e.g., one-hot).

```python
from sklekmeans import SSEKM
import numpy as np

X = np.random.rand(100, 2)
K = 3
prior = np.zeros((X.shape[0], K))
prior[:10, 0] = 1.0  # first 10 samples known to be in class 0

model = SSEKM(n_clusters=K, theta='auto', random_state=0)
model.fit(X, prior_matrix=prior)
print(model.cluster_centers_)
```

Documentation
-------------
The latest HTML documentation is hosted on GitHub Pages:

[ydcnanhe.github.io/sklearn-ekmeans](https://ydcnanhe.github.io/sklearn-ekmeans)

Badges above reflect build status; if the link 404s, wait for the docs CI to finish.

PyPI project page: https://pypi.org/project/sklekmeans/

Build and publish (maintainers)
-------------------------------
Local build of artifacts:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Publishing to PyPI is automated via GitHub Actions (Trusted Publishing). See `PUBLISHING.md`.

References
----------
- [1] Y. He. *An Equilibrium Approach to Clustering: Surpassing Fuzzy C-Means on Imbalanced Data*, IEEE Transactions on Fuzzy Systems, 2025.
- [2] Y. He. *Semi-supervised equilibrium K-means for imbalanced data clustering*, Knowledge-Based Systems, p.113990, 2025.
- [3] Y. He. *Imbalanced Data Clustering Using Equilibrium K-Means*, arXiv, 2024.

License
-------
BSD 3-Clause

