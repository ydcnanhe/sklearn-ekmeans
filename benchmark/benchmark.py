"""Monte Carlo benchmark: KMeans vs EKMeans on an imbalanced 3-cluster dataset.

This script repeatedly generates an imbalanced Gaussian mixture and compares
Adjusted Rand Index and Silhouette scores for scikit-learn KMeans and
Equilibrium K-Means (EKMeans). It also produces boxplots and a final
visualisation of true vs predicted labels for the last run.

Run:

    python benchmark/benchmark.py

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

N_REPEATS = 30  # Monte Carlo repetitions
N_SAMPLES = [2000, 50, 30]  # Strong imbalance
CENTERS = [(-5, -2), (0, 0), (5, 5)]
STD = [1.0, 1.0, 1.0]
N_CLUSTERS = 3
ari_results = {"KMeans": [], "EKM": []}
sil_results = {"KMeans": [], "EKM": []}
# ------------------
# Monte Carlo 实验
# ------------------
for seed in range(N_REPEATS):
    X, y_true = make_blobs(
        n_samples=N_SAMPLES,
        centers=CENTERS,
        cluster_std=STD,
        random_state=seed,
    )
    km = KMeans(n_clusters=N_CLUSTERS, n_init=30, random_state=seed)
    labels_km = km.fit_predict(X)
    ari_results["KMeans"].append(adjusted_rand_score(y_true, labels_km))
    sil_results["KMeans"].append(silhouette_score(X, labels_km))
    ekm = EKM(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
        alpha="dvariance",
        n_init=30,
        random_state=seed,
    )
    labels_ekm = ekm.fit_predict(X)
    ari_results["EKM"].append(adjusted_rand_score(y_true, labels_ekm))
    sil_results["EKM"].append(silhouette_score(X, labels_ekm))
# ------------------
# Statistical Summary
# ------------------
def stats(arr):
    return np.mean(arr), np.std(arr)
print("\n=== Monte Carlo Benchmark Results ({} runs) ===".format(N_REPEATS))
print("KMeans ARI      : {:.3f} ± {:.3f}".format(*stats(ari_results["KMeans"])))
print("EKM    ARI      : {:.3f} ± {:.3f}".format(*stats(ari_results["EKM"])))
print("KMeans Silhouette: {:.3f} ± {:.3f}".format(*stats(sil_results["KMeans"])))
print("EKM    Silhouette: {:.3f} ± {:.3f}".format(*stats(sil_results["EKM"])))
# ------------------
# Plot Boxplot
# ------------------
if _HAVE_PLT:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # ARI
    axs[0].boxplot(
        [ari_results["KMeans"], ari_results["EKM"]],
    labels=["KMeans", "EKM"], patch_artist=True,
    boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red")
)
    axs[0].set_title("Adjusted Rand Index (ARI) Distribution")
    axs[0].set_ylabel("ARI Score")
    # Silhouette
    axs[1].boxplot(
        [sil_results["KMeans"], sil_results["EKM"]],
        labels=["KMeans", "EKM"], patch_artist=True,
        boxprops=dict(facecolor="lightgreen"), medianprops=dict(color="red")
    )
    axs[1].set_title("Silhouette Distribution")
    axs[1].set_ylabel("Silhouette Score")
    plt.suptitle("Monte Carlo Benchmark ({} runs)".format(N_REPEATS), fontsize=14)
    plt.show()

# ------------------
# Cluster result visualization (last run data retained in loop scope)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    axes2[0].scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap='tab10', alpha=0.75)
    axes2[0].set_title('True Labels')
    axes2[1].scatter(X[:, 0], X[:, 1], c=labels_km, s=10, cmap='tab10', alpha=0.75)
    axes2[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='black', s=120, marker='X', edgecolor='white', linewidths=1)
    axes2[1].set_title('KMeans Clusters')
    axes2[2].scatter(X[:, 0], X[:, 1], c=labels_ekm, s=10, cmap='tab10', alpha=0.75)
    axes2[2].scatter(ekm.cluster_centers_[:, 0], ekm.cluster_centers_[:, 1], c='black', s=120, marker='X', edgecolor='white', linewidths=1)
    axes2[2].set_title('EKM Clusters')
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle('Cluster Result Visualization (Last Run)', fontsize=14)
    fig2.tight_layout()
    plt.show()
else:
    print("[Info] matplotlib not available; skipping plots.")
