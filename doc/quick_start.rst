Quick Start
===========

Installation
------------

Install from PyPI:

.. code-block:: bash

        pip install sklekmeans

Optional extras:

- With numba acceleration for speed:

    .. code-block:: bash

            pip install "sklekmeans[speed]"

- Development tools (tests, lint) and docs:

    .. code-block:: bash

            pip install "sklekmeans[dev]"
            pip install "sklekmeans[docs]"

Install and try EKMeans in a few lines:

.. code-block:: python

    from sklekmeans import EKMeans
    import numpy as np

    X = np.random.rand(200, 2)
    ekm = EKMeans(n_clusters=3, random_state=0, alpha='dvariance').fit(X)
    print(ekm.cluster_centers_)

See the :ref:`User Guide <ekmeans_user_guide>` for more details.
