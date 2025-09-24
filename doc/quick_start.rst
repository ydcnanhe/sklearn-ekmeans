Quick Start
===========

Install and try EKMeans in a few lines:

.. code-block:: python

    from sklekmeans import EKMeans
    import numpy as np

    X = np.random.rand(200, 2)
    ekm = EKMeans(n_clusters=3, random_state=0, alpha='dvariance').fit(X)
    print(ekm.cluster_centers_)

See the :ref:`User Guide <user_guide>` for more details.
