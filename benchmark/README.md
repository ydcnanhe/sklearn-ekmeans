# Benchmarks

This folder contains benchmarking scripts that explore performance and behavior of Equilibrium K-Means (EKMeans) under different settings.

## Available scripts

- `benchmark.py`: Monte Carlo comparison of KMeans vs EKMeans on a highly imbalanced low-dimensional Gaussian mixture. Reports ARI and Silhouette distributions and shows final clustering results.
- `benchmark_alphaSweep.py`: Sensitivity analysis scanning the `scale` parameter used in the `alpha='dvariance'` heuristic. Plots ARI and Silhouette versus scale alongside KMeans baselines.
- `benchmark_minibatch_compare.py`: Contrasts full-batch EKMeans with two mini-batch regimes: cumulative (accumulation) and online (exponential moving average) updates. Reports timing, ARI, NMI, internal objective estimate, cluster size distribution and effective epochs/iterations.
- `benchmark_dirichlet_highdim.py`: High-dimensional Dirichlet mixture benchmark generating imbalanced clusters with a controllable imbalance factor. Produces ARI, NMI, optional Silhouette (subsampled), SSE and timing statistics plus optional boxplots and 2D PCA projections.
- `benchmark_numba_ekm.py`: Measures wall-clock speed of EKMeans with and without numba JIT acceleration on a synthetic (optionally imbalanced) dataset and reports mean/std speed and approximate speedup factor.

## Running benchmarks

Install optional speed extras if you want numba acceleration benchmarking:

```bash
pip install -e .[speed]
```

Then run any script, for example:

```bash
python benchmark/benchmark_alphaSweep.py
```

For reproducibility each script exposes its own random seed handling or uses fixed seeds within loops.
