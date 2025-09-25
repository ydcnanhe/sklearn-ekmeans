"""
Imbalanced clustering comparison with EKMeans
=============================================

This example compares clustering performance on an imbalanced dataset
for several algorithms, including EKMeans and MiniBatchEKMeans.

It is intended for the gallery and requires matplotlib to render plots.

"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
    MeanShift,
    OPTICS,
    AffinityPropagation,
    Birch,
)
from sklearn.datasets import make_blobs

from sklekmeans import EKMeans, MiniBatchEKMeans


def _make_imbalanced(
    n_samples=2000,
    weights=(0.840, 0.01, 0.05),
    centers=np.array([[-3,-2],[2,-2],[2,2]]),
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
    for k in range(centers.shape[0]):
        idx = np.flatnonzero(y == k)
        take = int(round(weights[k] * n_samples))
        take = min(take, idx.size)
        sel = rng.choice(idx, size=take, replace=False)
        out_X.append(X[sel])
        out_y.append(np.full(sel.size, k))
    return np.vstack(out_X), np.concatenate(out_y)


def _plot(ax, X, labels, title, *, estimator=None, runtime=None):
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="tab10")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # Optional: annotate runtime in the bottom-right corner
    if runtime is not None:
        ax.text(
            0.99,
            0.01,
            (f"{runtime:.2f}s").lstrip("0"),
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
        )
    # If centroid-based, overlay centers
    if estimator is not None:
        centers = None
        if hasattr(estimator, "cluster_centers_") and estimator.cluster_centers_ is not None:
            centers = estimator.cluster_centers_
        elif hasattr(estimator, "cluster_centers_indices_") and estimator.cluster_centers_indices_ is not None:
            try:
                centers = X[np.asarray(estimator.cluster_centers_indices_, dtype=int)]
            except Exception:
                centers = None
        if centers is not None and centers.size > 0:
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                s=80,
                c="red",
                marker="x",
                linewidths=1.5,
            )


def main():
    X, y = _make_imbalanced()

    fig, axes = plt.subplots(3, 4, figsize=(12, 6), constrained_layout=True)
    axes = axes.ravel()
    _plot(axes[0], X, y, "Ground truth")

    t0 = time.perf_counter()
    ekm = EKMeans(n_clusters=3, n_init=10, random_state=0, alpha="dvariance").fit(X)
    t1 = time.perf_counter()
    _plot(axes[1], X, ekm.labels_, "EKMeans", estimator=ekm, runtime=(t1 - t0))

    t0 = time.perf_counter()
    mbekm = MiniBatchEKMeans(n_clusters=3, random_state=0, batch_size=256).fit(X)
    t1 = time.perf_counter()
    _plot(axes[2], X, mbekm.labels_, "MiniBatchEKMeans", estimator=mbekm, runtime=(t1 - t0))

    t0 = time.perf_counter()
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    t1 = time.perf_counter()
    _plot(axes[3], X, km.labels_, "KMeans", estimator=km, runtime=(t1 - t0))

    t0 = time.perf_counter()
    mbk = MiniBatchKMeans(n_clusters=3, n_init=10, random_state=0, batch_size=256).fit(X)
    t1 = time.perf_counter()
    _plot(axes[4], X, mbk.labels_, "MiniBatchKMeans", estimator=mbk, runtime=(t1 - t0))

    t0 = time.perf_counter()
    ms = MeanShift().fit(X)
    t1 = time.perf_counter()
    _plot(axes[5], X, ms.labels_, "MeanShift", estimator=ms, runtime=(t1 - t0))

    t0 = time.perf_counter()
    db = DBSCAN(min_samples=10).fit(X)
    t1 = time.perf_counter()
    _plot(axes[6], X, db.labels_, "DBSCAN", runtime=(t1 - t0))

    t0 = time.perf_counter()
    optics = OPTICS(min_samples=10).fit(X)
    t1 = time.perf_counter()
    _plot(axes[7], X, optics.labels_, "OPTICS", runtime=(t1 - t0))

    t0 = time.perf_counter()
    affinity_propagation = AffinityPropagation(random_state=0).fit(X)
    t1 = time.perf_counter()
    _plot(
        axes[8],
        X,
        affinity_propagation.labels_,
        "Affinity Propagation",
        estimator=affinity_propagation,
        runtime=(t1 - t0),
    )

    t0 = time.perf_counter()
    birch = Birch(n_clusters=3).fit(X)
    t1 = time.perf_counter()
    _plot(axes[9], X, birch.labels_, "Birch", estimator=birch, runtime=(t1 - t0))

    t0 = time.perf_counter()
    sc = SpectralClustering(n_clusters=3, assign_labels="kmeans", random_state=0).fit(X)
    t1 = time.perf_counter()
    _plot(axes[10], X, sc.labels_, "Spectral", runtime=(t1 - t0))

    t0 = time.perf_counter()
    ac = AgglomerativeClustering(n_clusters=3).fit(X)
    t1 = time.perf_counter()
    _plot(axes[11], X, ac.labels_, "Agglomerative", runtime=(t1 - t0))

    plt.show()


if __name__ == "__main__":
    main()
