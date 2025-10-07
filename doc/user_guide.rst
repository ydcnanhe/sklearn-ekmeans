.. _ekmeans_user_guide:

User Guide
==========

This project provides batch and mini-batch implementations of the Equilibrium
K-Means clustering objective. Both estimators follow the scikit-learn API.

Quick Start
-----------

.. code-block:: python

    from sklekmeans import EKMeans
    import numpy as np

    X = np.random.rand(100, 2)
    model = EKMeans(n_clusters=3, random_state=0)
    model.fit(X)
    print(model.cluster_centers_)

Mini-Batch Variant
------------------

.. code-block:: python

    from sklekmeans import MiniBatchEKMeans
    mb = MiniBatchEKMeans(n_clusters=3, random_state=0, batch_size=64)
    mb.fit(X)
    print(mb.cluster_centers_)

Semi-Supervised Variant (SSEKM)
-------------------------------

SSEKM extends EKMeans to incorporate partial labels or weak supervision via a
prior matrix (``prior_matrix`` or ``F``) of shape ``(n_samples, n_clusters)``:

- Rows with all zeros are considered unlabeled.
- A labeled row contains per-class probabilities (e.g., one-hot vectors).

Defaults and theta (supervision strength)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Alpha heuristic: By default ``alpha='dvariance'`` is used to set the equilibrium
    weighting based on data variance (same as EKMeans).
- Automatic: By default ``theta='auto'`` sets ``theta = |N| / |S|``, where
    ``|N|`` is the total number of samples and ``|S|`` is the number of labeled
    samples (rows with positive sum in ``F``). If ``|S| = 0``, then ``theta=0``
    and the method reduces to EKMeans.
- Numeric: You can pass a numeric ``theta`` to control the strength manually.
    It is used directly in both the supervised objective term and the labeled-row
    weight update.

Unlike a convex blend, here ``theta`` is not clipped to ``[0, 1]``. The weight
update for labeled rows uses

``W = W_ekm + theta * b * (F_norm - W_ekm)``,

where ``b`` is the mask of labeled rows and ``F_norm`` is the row-normalized
``F`` (labeled rows only). Larger ``theta`` makes the solution adhere more
strongly to provided labels; when ``theta=0`` it reduces to EKMeans.

Example:

.. code-block:: python

    import numpy as np
    from sklekmeans import SSEKM

    X = np.random.RandomState(0).randn(100, 2)
    K = 3
    # Construct prior matrix with partial labels (one-hot for a few samples)
    prior = np.zeros((X.shape[0], K), dtype=float)
    prior[:10, 0] = 1.0  # first 10 samples known to be in class 0

    model = SSEKM(n_clusters=K, theta=0.7, random_state=0)
    model.fit(X, prior_matrix=prior)
        print(model.cluster_centers_)
        # model.W_ are equilibrium weights adjusted by supervision for labeled rows
