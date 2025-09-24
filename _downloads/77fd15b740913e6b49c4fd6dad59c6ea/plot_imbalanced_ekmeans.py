"""
Imbalanced clustering comparison with EKMeans
=============================================

This example compares clustering performance on an imbalanced dataset
for several algorithms, including EKMeans and MiniBatchEKMeans.

It is intended for the gallery and requires matplotlib to render plots.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.datasets import make_blobs

from sklekmeans import EKMeans, MiniBatchEKMeans


def _make_imbalanced(
    n_samples=800,
    weights=(0.85, 0.10, 0.05),
    centers=3,
    cluster_std=(1.0, 1.0, 1.0),
    random_state=0,
):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    # Reweight labels to match desired imbalance by subsampling
    rng = np.random.RandomState(random_state)
    out_X, out_y = [], []
    for k in range(centers):
        idx = np.flatnonzero(y == k)
        take = int(round(weights[k] * n_samples))
        take = min(take, idx.size)
        sel = rng.choice(idx, size=take, replace=False)
        out_X.append(X[sel])
        out_y.append(np.full(sel.size, k))
    return np.vstack(out_X), np.concatenate(out_y)


def _plot(ax, X, labels, title):
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="tab10")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    X, y = _make_imbalanced()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    axes = axes.ravel()
    _plot(axes[0], X, y, "Ground truth")

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    _plot(axes[1], X, km.labels_, "KMeans")

    mbk = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=256).fit(X)
    _plot(axes[2], X, mbk.labels_, "MiniBatchKMeans")

    ekm = EKMeans(n_clusters=3, random_state=0, alpha="dvariance").fit(X)
    _plot(axes[3], X, ekm.labels_, "EKMeans")

    mbekm = MiniBatchEKMeans(n_clusters=3, random_state=0, batch_size=256).fit(X)
    _plot(axes[4], X, mbekm.labels_, "MiniBatchEKMeans")

    db = DBSCAN(eps=1.5, min_samples=10).fit(X)
    _plot(axes[5], X, db.labels_, "DBSCAN")

    sc = SpectralClustering(n_clusters=3, assign_labels="kmeans", random_state=0).fit(X)
    _plot(axes[6], X, sc.labels_, "Spectral")

    ac = AgglomerativeClustering(n_clusters=3).fit(X)
    _plot(axes[7], X, ac.labels_, "Agglomerative")

    plt.show()


if __name__ == "__main__":
    main()
