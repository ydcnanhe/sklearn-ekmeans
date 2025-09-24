"""
Basic usage of EKMeans and MiniBatchEKMeans
===========================================

This minimal example demonstrates fitting and using EKMeans and
MiniBatchEKMeans on random data.

Run with::

    python examples/example.py
"""

import numpy as np

from sklekmeans import EKMeans, MiniBatchEKMeans

# Generate synthetic data
X = np.random.rand(200, 2)
C_init = np.array([[1, 0], [0, 1]], dtype=np.float64)
model = EKMeans(n_clusters=2, metric="euclidean", alpha=0.5, init=C_init)
labels = model.fit_predict(X)
print("First 20 labels:", labels[:20])
print("First 5 membership rows:\n", model.U_[:5])
print("Cluster centers:\n", model.cluster_centers_)
print("Iterations:", model.n_iter_)
print("Objective J:", model.objective_)

X_new = np.random.rand(5, 2)
print("Predicted clusters for new samples:", model.predict(X_new))
print("Membership for new samples:\n", model.membership(X_new))
print("Distances for new samples:\n", model.transform(X_new))

print("\n=== MiniBatchEKMeans demo ===")
X_large = np.random.rand(2000, 2)
mb_model = MiniBatchEKMeans(
	n_clusters=2,
	metric="euclidean",
	alpha="dvariance",
	scale=2.0,
	batch_size=256,
	max_epochs=5,
	init="k-means++",
	shuffle=True,
	learning_rate=None,  # accumulation method
	tol=1e-3,
	reassign_patience=3,
	monitor_size=512,
	print_every=1,
	verbose=1,
	use_numba=False,
	random_state=0,
)
mb_model.fit(X_large)
print("Mini-batch centers:\n", mb_model.cluster_centers_)
print(
	"Approx objective last epoch:",
	mb_model.objective_approx_[-1] if mb_model.objective_approx_ else None,
)
