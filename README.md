sklekmeans - Equilibrium K-Means for scikit-learn
=================================================

![tests](https://github.com/ydcnanhe/sklekmeans/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/ydcnanhe/sklekmeans/graph/badge.svg)](https://codecov.io/gh/ydcnanhe/sklekmeans)
![doc](https://github.com/ydcnanhe/sklekmeans/actions/workflows/deploy-gh-pages.yml/badge.svg)
[![docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://ydcnanhe.github.io/sklearn-ekmeans)

`sklekmeans` provides batch and mini-batch implementations of the
Equilibrium K-Means (EKMeans) clustering algorithm. The method introduces
an equilibrium weighting scheme that can yield improved robustness on
imbalanced datasets compared to standard k-means.

Features
--------
* Drop-in scikit-learn compatible estimators: `EKMeans`, `MiniBatchEKMeans`.
* Supports Euclidean and Manhattan distances.
* Heuristic alpha selection via `alpha='dvariance'`.
* Mini-batch variant with accumulation or online update modes.
* Soft memberships (`membership`) and equilibrium weights (`W_`).

Installation
------------
The project is not yet published on PyPI, so `pip install sklekmeans` will fail with
"No matching distribution". Install from source instead:

1. Clone the repository:
```bash
git clone https://github.com/ydcnanhe/sklearn-ekmeans.git
cd sklearn-ekmeans
```
2. (Recommended) Create a fresh virtual environment.
3. Choose one of the following install modes:

Base (runtime only):
```bash
pip install -e .
```

With development tools (tests, lint):
```bash
pip install -e .[dev]
```

With docs build deps:
```bash
pip install -e .[docs]
```

With optional numba acceleration:
```bash
pip install -e .[speed]
```

Everything (dev + docs + speed):
```bash
pip install -e .[all]
```

Alternatively you can use the provided requirement files:
```bash
pip install -r requirements.txt            # base
pip install -r requirements-dev.txt        # full stack (includes base)
```

Install directly from GitHub (no clone) specifying extras, e.g. full stack:
```bash
pip install "git+https://github.com/ydcnanhe/sklearn-ekmeans.git#egg=sklekmeans[all]"
```

If you later publish to PyPI, the simple form will become:
```bash
pip install sklekmeans[speed]
```

Quick Start
-----------
```python
from sklekmeans import EKMeans
import numpy as np

X = np.random.rand(200, 2)
ekm = EKMeans(n_clusters=3, random_state=0, alpha='dvariance').fit(X)
print(ekm.cluster_centers_)
```

Documentation
-------------
The latest HTML documentation is hosted on GitHub Pages:

[ydcnanhe.github.io/sklearn-ekmeans](https://ydcnanhe.github.io/sklearn-ekmeans)

Badges above reflect build status; if the link 404s, wait for the docs CI to finish.

References
----------
- [1] Y. He. *An Equilibrium Approach to Clustering: Surpassing Fuzzy C-Means on Imbalanced Data*, IEEE Transactions on Fuzzy Systems, 2025.
- [2] Y. He. *Imbalanced Data Clustering Using Equilibrium K-Means*, arXiv, 2024.

License
-------
BSD 3-Clause

