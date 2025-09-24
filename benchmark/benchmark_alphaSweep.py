"""Alpha sensitivity benchmark for EKMeans.

This script sweeps the ``scale`` parameter that influences the
heuristic ``alpha`` value when ``alpha='dvariance'`` and compares the
Adjusted Rand Index (ARI) and Silhouette score of EKMeans against the
baseline scikit-learn KMeans under an imbalanced 3-cluster setting.

Run:

    python benchmark/benchmark_alphaSweep.py

Outputs summary statistics and plots mean metric values vs the
``scale`` parameter.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

from sklekmeans import EKMeans as EKM

try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

N_REPEATS = 20  # Monte Carlo repetitions
N_SAMPLES = [2000, 50, 30]  # Strong class imbalance
CENTERS = [(-5, -2), (0, 0), (5, 5)]
STD = [1.0, 1.0, 1.0]
N_CLUSTERS = 3
scale_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

results_ekm = {scale: {"ARI": [], "Silhouette": []} for scale in scale_list}
results_km = {"ARI": [], "Silhouette": []}

for seed in range(N_REPEATS):
    X, y_true = make_blobs(
        n_samples=N_SAMPLES, centers=CENTERS, cluster_std=STD, random_state=seed
    )
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
    labels_km = km.fit_predict(X)
    results_km["ARI"].append(adjusted_rand_score(y_true, labels_km))
    results_km["Silhouette"].append(silhouette_score(X, labels_km))
    for scale in scale_list:
        ekm = EKM(
            n_clusters=N_CLUSTERS,
            metric="euclidean",
            alpha="dvariance",
            scale=scale,
            n_init=10,
            random_state=seed,
        )
        labels_ekm = ekm.fit_predict(X)
        results_ekm[scale]["ARI"].append(adjusted_rand_score(y_true, labels_ekm))
        results_ekm[scale]["Silhouette"].append(silhouette_score(X, labels_ekm))


def stats(arr):
    return np.mean(arr), np.std(arr)


print(f"\n=== Benchmark Results ({N_REPEATS} runs) ===")
print("KMeans ARI       : {:.3f} ± {:.3f}".format(*stats(results_km["ARI"])))
print(
    "KMeans Silhouette: {:.3f} ± {:.3f}".format(
        *stats(results_km["Silhouette"])
    )
)
for scale in scale_list:
    m_ari, s_ari = stats(results_ekm[scale]["ARI"])
    m_sil, s_sil = stats(results_ekm[scale]["Silhouette"])
    print(
        f"EKM[scale={scale}] ARI: {m_ari:.3f} ± {s_ari:.3f}, Silhouette: {m_sil:.3f} ± {s_sil:.3f}"
    )

num_scales = scale_list
ari_means = [np.mean(results_ekm[a]["ARI"]) for a in num_scales]
sil_means = [np.mean(results_ekm[a]["Silhouette"]) for a in num_scales]

if _HAVE_PLT:
    plt.figure(figsize=(10, 5))
    plt.plot(num_scales, ari_means, marker="o", label="EKM ARI")
    plt.plot(num_scales, sil_means, marker="s", label="EKM Silhouette")
    plt.axhline(
        np.mean(results_km["ARI"]),
        color="red",
        linestyle="--",
        label="KMeans ARI (baseline)",
    )
    plt.axhline(
        np.mean(results_km["Silhouette"]),
        color="green",
        linestyle="--",
        label="KMeans Silhouette (baseline)",
    )
    plt.xlabel("Scale parameter (affects alpha heuristic)")
    plt.ylabel("Score (mean)")
    plt.title(f"EKM Sensitivity to Scale (mean over {N_REPEATS} runs)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print('[Info] matplotlib not installed; only text output without plots will be shown.')
