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
