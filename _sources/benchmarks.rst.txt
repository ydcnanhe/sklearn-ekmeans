Benchmarks
==========

The repository provides several benchmarking scripts under the
``benchmark/`` directory illustrating different aspects of the
Equilibrium K-Means implementation.

Available Scripts
-----------------

``benchmark.py``
    Monte Carlo comparison of KMeans vs EKMeans on a highly imbalanced
    low-dimensional Gaussian mixture. Reports ARI and Silhouette
    distributions and shows final clustering results.

``benchmark_alphaSweep.py``
    Sensitivity analysis scanning the ``scale`` parameter used in the
    ``alpha='dvariance'`` heuristic. Plots ARI and Silhouette versus
    scale alongside KMeans baselines.

``benchmark_minibatch_compare.py``
    Contrasts full-batch EKMeans with two mini-batch regimes: cumulative
    (accumulation) and online (exponential moving average) updates.
    Reports timing, ARI, NMI, internal objective estimate, cluster size
    distribution and effective epochs/iterations.

``benchmark_dirichlet_highdim.py``
    High-dimensional Dirichlet mixture benchmark generating imbalanced
    clusters with a controllable imbalance factor. Produces ARI, NMI,
    optional Silhouette (subsampled), SSE and timing statistics plus
    optional boxplots and 2D PCA projections.

``benchmark_numba_ekm.py``
    Measures wall-clock speed of EKMeans with and without numba JIT
    acceleration on a synthetic (optionally imbalanced) dataset and
    reports mean/std speed and approximate speedup factor.

Running Benchmarks
------------------

Install optional speed extras if you want numba acceleration benchmarking:

.. code-block:: bash

    pip install -e .[speed]

Then run any script, for example:

.. code-block:: bash

    python benchmark/benchmark_alphaSweep.py

For reproducibility each script exposes its own random seed handling or
uses fixed seeds within loops.
