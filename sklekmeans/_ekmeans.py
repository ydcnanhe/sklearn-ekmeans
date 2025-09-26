"""Equilibrium K-Means clustering algorithms.

This module implements Equilibrium K-Means (EKMeans) and a mini-batch
variant (:class:`MiniBatchEKMeans`). The objective modifies standard
k-means by introducing an equilibrium weighting scheme controlled by a
parameter ``alpha`` that can improve robustness on imbalanced datasets.

The implementation follows the structure of scikit-learn's k-means
estimators where practical, but the internal update rules differ.

References
----------

.. [1] Y. He. *An Equilibrium Approach to Clustering: Surpassing Fuzzy
   C-Means on Imbalanced Data*, IEEE Transactions on Fuzzy Systems,
   2025.
.. [2] Y. He. *Imbalanced Data Clustering Using Equilibrium K-Means*,
   arXiv, 2024.

Notes
-----
Both estimators expose helper methods ``membership`` and
``fit_membership`` that return soft assignment matrices derived from the
exponential weighting prior to equilibrium correction. The equilibrium
weight matrix ``W_`` is also stored after fitting.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, _fit_context
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

# Optional numba acceleration (soft dependency)
try:  # pragma: no cover - optional path
    from numba import njit, prange, set_num_threads  # type: ignore

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional path
    _NUMBA_AVAILABLE = False

###############################################################################
# Helper utilities


def _pairwise_distance(X, Y=None, metric: str = "euclidean"):
    if metric == "euclidean":
        return euclidean_distances(X, Y, squared=False)
    if metric == "manhattan":
        return manhattan_distances(X, Y)
    raise ValueError(f"Unsupported distance metric: {metric!r}")


def _kmeans_plus_like(X, n_clusters, *, metric="euclidean", random_state=None):
    """Lightweight k-means++ style initialisation supporting non-euclidean.

    Falls back to metric-based sampling when the compiled kmeans++ is not
    applicable (e.g. Manhattan distance).
    """
    rng = check_random_state(random_state)
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=float)
    # choose first center uniformly
    first = rng.randint(n_samples)
    centers[0] = X[first]
    for k in range(1, n_clusters):
        D2 = np.min(_pairwise_distance(X, centers[:k], metric) ** 2, axis=1)
        total = np.sum(D2)
        if not np.isfinite(total) or total <= 0:
            idx = rng.randint(n_samples)
        else:
            probs = D2 / total
            idx = rng.choice(n_samples, p=probs)
        centers[k] = X[idx]
    return centers


def _calc_weight_numpy(D2, alpha):
    # D2: (n_samples, n_clusters) assumed shifted per row for stability
    E = np.exp(-alpha * D2)
    U = E / np.sum(E, axis=1, keepdims=True)
    W = U * (1 - alpha * (D2 - np.sum(D2 * U, axis=1, keepdims=True)))
    # safeguard degenerate rows
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    for i in zero_idx:
        pos = np.argmin(D2[i])
        W[i] = 0.0
        W[i, pos] = 1.0
    return W


if _NUMBA_AVAILABLE:  # pragma: no cover - exercised only when numba present

    @njit(parallel=True, fastmath=True)
    def _calc_weight_numba(D2, alpha):  # type: ignore
        n_samples, n_clusters = D2.shape
        W = np.empty((n_samples, n_clusters), dtype=np.float64)
        for i in prange(n_samples):
            sum_e = 0.0
            sum_d2e = 0.0
            row_e = np.empty(n_clusters, dtype=np.float64)
            for k in range(n_clusters):
                e = np.exp(-alpha * D2[i, k])
                row_e[k] = e
                sum_e += e
                sum_d2e += D2[i, k] * e
            denom = sum_e
            J_i = sum_d2e / denom
            row_sumW = 0.0
            for k in range(n_clusters):
                w = (row_e[k] / denom) * (1.0 - alpha * (D2[i, k] - J_i))
                W[i, k] = w
                row_sumW += w
            if row_sumW == 0.0:
                best = D2[i, 0]
                pos = 0
                for k in range(1, n_clusters):
                    if D2[i, k] < best:
                        best = D2[i, k]
                        pos = k
                for k in range(n_clusters):
                    W[i, k] = 0.0
                W[i, pos] = 1.0
        return W


###############################################################################
# Core estimator: Equilibrium K-Means


class EKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """Equilibrium K-Means clustering.

    A robust variant of k-means designed for imbalanced datasets. The
    method uses an equilibrium weighting scheme parameterised by
    ``alpha``. For ``alpha='dvariance'`` a heuristic based on the data
    variance is used.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids
        to generate.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric used both to assign points to clusters and to
        update centers. Manhattan distance can be more robust to outliers
        in some settings but increases cost relative to vectorised
        squared Euclidean computations.
    alpha : float or {'dvariance'}, default=0.5
        Equilibrium weighting parameter. If set to
        the string ``'dvariance'`` a heuristic value ``scale / mean(d^2)``
        is computed where ``d^2`` are squared distances to the global
        mean.
    scale : float, default=2.0
        Multiplicative factor applied in the ``'dvariance'`` heuristic.
        Higher values yield larger effective ``alpha`` resulting in
        crisper assignments.
    max_iter : int, default=300
        Maximum number of EM-like update iterations for a single
        initialisation.
    tol : float, default=1e-4
        Relative tolerance (scaled by average feature variance of the
        data) on the Frobenius norm of the change in ``cluster_centers_``
        to declare convergence.
    n_init : int, default=1
        Number of random initialisations to perform. The run with the
        lowest internal equilibrium objective is retained. Increasing
        ``n_init`` improves robustness to local minima at additional
        computational cost.
    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features), default='k-means++'
        Method for initialization.
        * 'k-means++' : use a probabilistic seeding adapted for the
          chosen metric.
        * 'random' : choose ``n_clusters`` observations at random.
        * ndarray : user provided initial centers.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of initial center selection and the
        heuristic alpha sampling (when applicable). Pass an int for
        reproducible results.
    use_numba : bool, default=False
        If ``True`` and :mod:`numba` is installed (see ``[speed]`` extra),
        use a JIT-compiled kernel for weight computation.
    numba_threads : int or None, default=None
        If provided sets the number of threads used by numba parallel
        sections. Ignored if numba is unavailable or ``use_numba`` is
        ``False``.
    verbose : int, default=0
        Verbosity level. ``0`` is silent; higher values print progress
        each iteration.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Hard assignment labels for training data.
    n_iter_ : int
        Number of iterations run for the best initialisation.
    objective_ : float
        Objective value of the best run.
    alpha_ : float
        Resolved alpha value actually used.
    W_ : ndarray of shape (n_samples, n_clusters)
        Equilibrium weights after fitting.
    U_ : ndarray of shape (n_samples, n_clusters)
        Membership matrix (soft assignments based on exp(-alpha * d^2)).
        Each row sums to 1.
    n_features_in_ : int
        Number of features seen during :meth:`fit`. Set by the first call to
        :meth:`fit` and used for input validation in subsequent operations.

    Methods
    -------
    fit(X, y=None)
        Fit the model and learn cluster centers.
    predict(X)
        Return the hard cluster label (nearest center) for each sample.
    transform(X)
        Return matrix of distances from samples to cluster centers.
    fit_predict(X, y=None)
        Fit the model and return training labels in one pass.
    membership(X)
        Compute soft membership (row-normalized responsibilities).
    fit_membership(X, y=None)
        Fit the model and return the training membership matrix.

    (Internal helpers: `_resolve_alpha`, `_init_centers`, `_calc_weight`, `_objective` are internal and not public API.)

    Notes
    -----
    The average complexity is roughly :math:`O(k^2 n T)` due to the
    weight update per iteration, where ``k`` is the number of clusters,
    ``n`` the number of samples and ``T`` the number of iterations. The
    algorithm can fall into local minima; using ``n_init>1`` is
    recommended.

    Examples
    --------

    >>> from sklekmeans import EKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> ekmeans = EKMeans(n_clusters=2, random_state=0, n_init=1).fit(X)
    >>> ekmeans.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> ekmeans.predict([[0, 0], [12, 3]])
    array([1, 0])
    >>> ekmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """

    _parameter_constraints = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions({"euclidean", "manhattan"})],
        "alpha": [Real, StrOptions({"dvariance"})],
        "scale": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), np.ndarray],
        "random_state": [None, Integral],
        "use_numba": [bool],
        "numba_threads": [None, Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        metric="euclidean",
        alpha=0.5,
        scale=2.0,
        max_iter=300,
        tol=1e-4,
        n_init=1,
        init="k-means++",
        random_state=None,
        use_numba=False,
        numba_threads=None,
        verbose=0,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.alpha = alpha
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self.use_numba = use_numba
        self.numba_threads = numba_threads
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _resolve_alpha(self, X):
        """Resolve the effective alpha value.

        If the user specified a float, return it directly. If the user
        specified the string 'dvariance', compute a heuristic value based
        on the mean squared distance to the global mean scaled by
        ``scale``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        alpha : float
            Effective numeric alpha.
        """
        alpha = self.alpha
        if isinstance(alpha, str):
            if alpha == "dvariance":
                mu = np.mean(X, axis=0, keepdims=True)
                d2 = _pairwise_distance(X, mu, self.metric) ** 2
                dv = float(np.mean(d2))
                alpha = self.scale / max(dv, np.finfo(float).eps)
            else:  # pragma: no cover - guarded by param validation
                raise ValueError("Unsupported alpha string.")
        return float(alpha)

    def _init_centers(self, X, rng):
        """Initialise cluster centers according to the chosen strategy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix.
        rng : RandomState
            Random generator instance.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial cluster centers.
        """
        if isinstance(self.init, np.ndarray):
            centers = np.asarray(self.init, dtype=float)
            if centers.shape[0] != self.n_clusters:
                raise ValueError(
                    "init array should have shape (n_clusters, n_features)"
                )
            return centers.copy()
        if self.init == "k-means++":
            if self.metric == "euclidean":
                centers, _ = kmeans_plusplus(
                    X, self.n_clusters, random_state=rng, sample_weight=None
                )
                centers = centers.astype(float, copy=False)
            else:
                centers = _kmeans_plus_like(
                    X, self.n_clusters, metric=self.metric, random_state=rng
                )
        elif self.init == "random":
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            centers = X[idx].astype(float, copy=False)
        else:  # pragma: no cover - guarded by param validation
            raise ValueError("Unknown init method")
        return centers

    def _calc_weight(self, D2, alpha):
        """Compute equilibrium weights for a squared distance matrix.

        Parameters
        ----------
        D2 : ndarray of shape (n_samples, n_clusters)
            Row-shifted squared distances to cluster centers.
        alpha : float
            Weighting parameter.

        Returns
        -------
        W : ndarray of shape (n_samples, n_clusters)
            Equilibrium weights (can contain zeros, not necessarily rows summing to 1).
        """
        if self.use_numba and _NUMBA_AVAILABLE:
            if self.numba_threads is not None:
                try:
                    set_num_threads(int(self.numba_threads))  # type: ignore
                except Exception:  # pragma: no cover
                    pass
            return _calc_weight_numba(D2, alpha)  # type: ignore
        return _calc_weight_numpy(D2, alpha)

    def _objective(self, X, cluster_centers, alpha):
        D2 = _pairwise_distance(X, cluster_centers, self.metric) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        E = np.exp(-alpha * D2_shift)
        U = E / np.sum(E, axis=1, keepdims=True)
        obj = np.sum(U * D2)
        return obj

    # ------------------------------------------------------------------
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute Equilibrium K-Means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=True,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        K = self.n_clusters
        alpha = self._resolve_alpha(X)
        verbose = self.verbose
        Var_X = np.mean(np.var(X, axis=0))

        best_obj = np.inf
        best_centers = None
        best_labels = None
        best_niter = None

        for run in range(self.n_init):
            centers = self._init_centers(X, rng)
            if verbose:
                print("Initialization complete")
            prev = centers.copy()
            for it in range(self.max_iter):
                D2 = _pairwise_distance(X, centers, self.metric) ** 2
                D2_shift = D2 - D2.min(axis=1, keepdims=True)
                W = self._calc_weight(D2_shift, alpha)
                for k in range(K):
                    sw = np.maximum(np.sum(W[:, k]), np.finfo(float).eps)
                    centers[k] = (W[:, k] @ X) / sw

                if verbose:
                    obj = self._objective(X, centers, alpha)
                    print(f"Iteration {it}, objective/loss {obj:<.3e}.")

                center_shift_tot = np.linalg.norm(centers - prev, "fro")
                if self.tol > 0.0 and center_shift_tot <= Var_X * self.tol:
                    if verbose:
                        print(
                            f"Converged at iteration {it} (center shift / var(X) "
                            f"{center_shift_tot / Var_X:<.3e} <= tol {self.tol:<.3e})."
                        )
                    break
                else:
                    if verbose and it == self.max_iter - 1:
                        print(
                            f"Reached max_iter {self.max_iter} (center shift / var(X)"
                            f"{center_shift_tot / Var_X:<.3e} > tol {self.tol:<.3e})."
                        )
                prev = centers.copy()
            # evaluate objective
            D2_eval = _pairwise_distance(X, centers, self.metric) ** 2
            obj = self._objective(X, centers, alpha)
            if obj < best_obj:
                best_obj = obj
                best_centers = centers
                best_labels = np.argmin(D2_eval, axis=1)
                best_niter = it

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.n_iter_ = best_niter
        self.objective_ = float(best_obj)
        self.alpha_ = alpha

        # cache distances and weights for training data
        D = _pairwise_distance(X, self.cluster_centers_, self.metric)
        D2 = D**2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        self.W_ = self._calc_weight(D2_shift, alpha)
        E = np.exp(-alpha * D2_shift)
        self.U_ = E / np.sum(E, axis=1, keepdims=True)
        return self

    def predict(self, X):
        """Predict the closest cluster index for each sample in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the closest learned cluster center for each sample.
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        D = _pairwise_distance(X, self.cluster_centers_, self.metric)
        labels = np.argmin(D, axis=1)
        distinct_clusters = len(set(labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )
        return labels

    def transform(self, X):
        """Compute distances of samples to each cluster center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        distances : ndarray of shape (n_samples, n_clusters)
            Pairwise distances to `cluster_centers_` using the configured metric.
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return _pairwise_distance(X, self.cluster_centers_, self.metric)

    def fit_predict(self, X, y=None):
        """Fit the model to ``X`` and return cluster indices.

        Equivalent to calling ``fit(X)`` followed by ``predict(X)`` but
        more efficient.
        """
        return self.fit(X, y).labels_

    def membership(self, X):
        """Return membership (soft assignment) matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        U : ndarray of shape (n_samples, n_clusters)
            Row-stochastic soft assignment matrix (rows sum to 1).
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        alpha = self.alpha_
        D2 = self.transform(X) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        E = np.exp(-alpha * D2_shift)
        U = E / np.sum(E, axis=1, keepdims=True)
        return U

    def fit_membership(self, X, y=None):
        """Fit the model and return the membership matrix for training data."""
        return self.fit(X, y).U_


###############################################################################
# Mini-batch variant


class MiniBatchEKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """Mini-batch Equilibrium K-Means.

    Scalable mini-batch optimisation of the equilibrium k-means objective
    supporting both an accumulation scheme (``learning_rate=None``) and
    an online exponential moving average update scheme.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form / centers to learn.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric used for batch distance computations. Manhattan
        can be more robust to certain outliers but is slower than
        vectorised Euclidean.
    alpha : float or {'dvariance'}, default=0.5
        Equilibrium weighting parameter controlling sharpness of the
        soft membership distribution prior to equilibrium correction. If
        ``'dvariance'`` a heuristic value is derived from a subsample of
        the data (see ``init_size``) using ``scale / mean(d^2)`` where
        ``d^2`` are squared distances to the subsample mean.
    scale : float, default=2.0
        Multiplicative factor in the ``'dvariance'`` heuristic. Larger
        values produce larger effective ``alpha`` leading to crisper
        initial memberships.
    batch_size : int, default=256
        Number of samples per mini-batch. A larger batch size reduces
        variance of updates but increases per-step cost and memory.
    max_epochs : int, default=10
        Maximum number of full passes (epochs) over the training data.
    n_init : int, default=1
        Number of random initialisations. The algorithm will run
        mini-batch optimisation ``n_init`` times with different seeds
        (derived from ``random_state``) and keep the run with the lowest
        internal equilibrium objective (evaluated on the full dataset),
        which improves robustness to local minima.
    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features), default='k-means++'
        Initialization method.
        * 'k-means++' : probabilistic seeding adapted for chosen metric.
        * 'random' : choose ``n_clusters`` observations without replacement.
        * ndarray : user-specified initial centers.
    init_size : int or None, default=None
        Subsample size used to estimate the ``'dvariance'`` heuristic. If
        ``None`` a size based on ``max(10 * n_clusters, batch_size)`` is
        used (capped at ``n_samples``). Ignored when ``alpha`` is a
        numeric value.
    shuffle : bool, default=True
        Whether to shuffle sample order at the beginning of each epoch.
        Recommended for i.i.d. data to decorrelate batches.
    learning_rate : float or None, default=None
        If ``None`` use accumulation mode (centers are the weighted
        average of all processed batches). If a positive float, perform
        online exponential moving average updates::

            C_k <- (1 - lr) * C_k + lr * xbar_k

        where ``xbar_k`` is the weighted mean of cluster ``k`` in the
        current batch.
    tol : float, default=1e-4
        Convergence tolerance on Frobenius norm of center change scaled
        by average feature variance of the dataset (computed once).
    reassignment_ratio : float, default=0.0
        Minimum fraction of batch weight a cluster must receive to be
        updated. Clusters not meeting the threshold accumulate a
        patience counter (see ``reassign_patience``).
    reassign_patience : int, default=3
        Number of consecutive batches a cluster can fail the
        ``reassignment_ratio`` threshold before it is forcibly
        reassigned to a far point in the current batch.
    verbose : int, default=0
        Verbosity level. ``0`` is silent; higher values print epoch
        diagnostics every ``print_every`` epochs.
    monitor_size : int or None, default=1024
        Size of a subsample used to compute an approximate objective for
        monitoring (stored in ``objective_approx_``). If ``None`` the
        full dataset is used (higher cost).
    print_every : int, default=1
        Frequency (in epochs) at which progress messages are printed
        when ``verbose > 0``.
    use_numba : bool, default=False
        If ``True`` and :mod:`numba` is installed use a JIT-compiled
        kernel for the equilibrium weight computation.
    numba_threads : int or None, default=None
        Number of threads to request from numba's threading layer (if
        available). Ignored when numba is not installed or
        ``use_numba=False``.
    random_state : int, RandomState instance or None, default=None
        Controls reproducibility of center initialisation, alpha
        heuristic subsampling and shuffling. Pass an int for
        deterministic behaviour.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final centers.
    labels_ : ndarray of shape (n_samples,)
        Hard assignment labels for the training data (available only after
        calling :meth:`fit`). Not updated by :meth:`partial_fit`.
    alpha_ : float
        Resolved alpha value.
    objective_approx_ : list of float
        Epoch-wise approximate objectives.
    counts_ : ndarray of shape (n_clusters,)
        Accumulated weights (accumulation mode).
    sums_ : ndarray of shape (n_clusters, n_features)
        Accumulated weighted sums (accumulation mode).
    W_, U_ : ndarrays
        Final weights and memberships for the full training data (if `fit`).
    n_features_in_ : int
        Number of features seen during the first call to :meth:`fit` or
        :meth:`partial_fit`. Ensures consistent dimensionality across
        incremental updates and predictions.

    Methods
    -------
    fit(X, y=None)
        Run full mini-batch training until convergence or max epochs.
    partial_fit(X_batch, y=None)
        Update model parameters using a single mini-batch.
    predict(X)
        Return hard cluster labels for samples.
    transform(X)
        Return distances from samples to cluster centers.
    membership(X)
        Compute soft membership (row-normalized responsibilities).
    fit_predict(X, y=None)
        Fit the model and return hard labels for X.
    fit_membership(X, y=None)
        Fit the model and return the membership matrix for the training data.

    (Internal helpers: `_init_centers`, `_resolve_alpha`, `_calc_weight`, `_approx_objective` are internal implementation details.)
    Notes
    -----
    The approximate objective is tracked on a monitoring subset when
    ``monitor_size`` is not ``None`` and stored in
    ``objective_approx_``.
    Examples
    --------
    >>> from sklekmeans import MiniBatchEKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> ekmeans = MiniBatchEKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6)
    >>> ekmeans = ekmeans.partial_fit(X[0:6,:])
    >>> ekmeans = ekmeans.partial_fit(X[6:12,:])
    >>> ekmeans.cluster_centers_
    array([[3.47914144, 3.02885195],
          [0.73800796, 0.61514045]])
    >>> ekmeans.predict([[0, 0], [4, 4]])
    array([1, 0])
    >>> # fit on the whole data
    >>> ekmeans = MiniBatchEKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_epochs=10).fit(X)
    >>> ekmeans.cluster_centers_
    array([[3.51549642, 4.53433897],
       [1.98848002, 0.97403648]])
    >>> ekmeans.predict([[0, 0], [4, 4]])
    array([1, 0])
    """

    _parameter_constraints = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions({"euclidean", "manhattan"})],
        "alpha": [Real, StrOptions({"dvariance"})],
        "scale": [Interval(Real, 0, None, closed="neither")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "max_epochs": [Interval(Integral, 1, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), np.ndarray],
        "init_size": [None, Interval(Integral, 1, None, closed="left")],
        "shuffle": [bool],
        "learning_rate": [None, Interval(Real, 0, None, closed="neither")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "reassignment_ratio": [Interval(Real, 0, 1, closed="both")],
        "reassign_patience": [Interval(Integral, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left")],
        "monitor_size": [None, Interval(Integral, 1, None, closed="left")],
        "print_every": [Interval(Integral, 1, None, closed="left")],
        "use_numba": [bool],
        "numba_threads": [None, Interval(Integral, 1, None, closed="left")],
        "random_state": [None, Integral],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        metric="euclidean",
        alpha=0.5,
        scale=2.0,
        batch_size=256,
        max_epochs=10,
        n_init=1,
        init="k-means++",
        init_size=None,
        shuffle=True,
        learning_rate=None,
        tol=1e-4,
        reassignment_ratio=0.0,
        reassign_patience=3,
        verbose=0,
        monitor_size=1024,
        print_every=1,
        use_numba=False,
        numba_threads=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.alpha = alpha
        self.scale = scale
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_init = n_init
        self.init = init
        self.init_size = init_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.tol = tol
        self.reassignment_ratio = reassignment_ratio
        self.reassign_patience = reassign_patience
        self.verbose = verbose
        self.monitor_size = monitor_size
        self.print_every = print_every
        self.use_numba = use_numba
        self.numba_threads = numba_threads
        self.random_state = random_state

    # ---------------- internal helpers ----------------
    def _init_centers(self, X, rng):
        """Initialise centers for mini-batch optimisation."""
        if isinstance(self.init, np.ndarray):
            return np.asarray(self.init, dtype=float).copy()
        if self.init == "k-means++":
            if self.metric == "euclidean":
                centers, _ = kmeans_plusplus(
                    X, self.n_clusters, random_state=rng, sample_weight=None
                )
                centers = centers.astype(float, copy=False)
            else:
                centers = _kmeans_plus_like(
                    X, self.n_clusters, metric=self.metric, random_state=rng
                )
        elif self.init == "random":
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            centers = X[idx].astype(float, copy=False)
        else:  # pragma: no cover
            raise ValueError("Unsupported init method")
        return centers

    def _resolve_alpha(self, X, rng):
        """Resolve alpha (supports heuristic when alpha='dvariance')."""
        alpha = self.alpha
        if isinstance(alpha, str):
            if alpha == "dvariance":
                if self.init_size is None:
                    n0 = min(
                        X.shape[0],
                        max(10 * self.n_clusters, self.batch_size),
                    )
                else:
                    n0 = min(X.shape[0], self.init_size)
                idx = rng.choice(X.shape[0], size=n0, replace=False)
                X0 = X[idx]
                mu = np.mean(X0, axis=0, keepdims=True)
                d2 = _pairwise_distance(X0, mu, self.metric) ** 2
                dv = float(np.mean(d2))
                alpha = self.scale / max(dv, np.finfo(float).eps)
            else:  # pragma: no cover
                raise ValueError("Unsupported alpha option")
        return float(alpha)

    def _calc_weight(self, D2, alpha):
        """Compute equilibrium weights for a batch distance matrix."""
        if self.use_numba and _NUMBA_AVAILABLE:
            if self.numba_threads is not None:
                try:
                    set_num_threads(int(self.numba_threads))  # type: ignore
                except Exception:  # pragma: no cover
                    pass
            return _calc_weight_numba(D2, alpha)  # type: ignore
        return _calc_weight_numpy(D2, alpha)

    def _approx_objective(self, Xs, centers, alpha):
        """Approximate objective on a monitoring subset Xs."""
        D2 = _pairwise_distance(Xs, centers, self.metric) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        E = np.exp(-alpha * D2_shift)
        U = E / np.sum(E, axis=1, keepdims=True)
        return float(np.sum(U * D2))

    # ---------------- public API ----------------
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Train the mini-batch equilibrium k-means estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            For API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=True,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        K = self.n_clusters
        Var_X = np.mean(np.var(X, axis=0))
        alpha = self._resolve_alpha(X, rng)

        best_obj = np.inf
        best_centers = None
        best_alpha = None
        best_epoch = None
        best_counts = None
        best_sums = None
        best_obj_approx_hist = None

        for run in range(self.n_init):
            centers = self._init_centers(X, rng)

            empty_counts = np.zeros(K, dtype=np.int64)
            prev = centers.copy()
            obj_approx_hist = []

            if self.monitor_size is None:
                monitor_idx = np.arange(n_samples)
            else:
                ms = min(n_samples, self.monitor_size)
                monitor_idx = rng.choice(n_samples, size=ms, replace=False)

            Nk = np.zeros(K, dtype=float)
            Sk = np.zeros((K, n_features), dtype=float)

            for epoch in range(1, self.max_epochs + 1):
                if self.shuffle:
                    order = rng.permutation(n_samples)
                else:
                    order = np.arange(n_samples)

                for start in range(0, n_samples, self.batch_size):
                    end = min(start + self.batch_size, n_samples)
                    batch_idx = order[start:end]
                    Xb = X[batch_idx]

                    D2 = _pairwise_distance(Xb, centers, self.metric) ** 2
                    D2_shift = D2 - D2.min(axis=1, keepdims=True)
                    W = self._calc_weight(D2_shift, alpha)
                    wk_sum = W.sum(axis=0)
                    update_mask = wk_sum > self.reassignment_ratio * Xb.shape[0]

                    for k in range(K):
                        if update_mask[k]:
                            empty_counts[k] = 0
                        else:
                            empty_counts[k] += 1

                    if self.learning_rate is None:
                        if not np.all(update_mask):
                            W_eff = W.copy()
                            W_eff[:, ~update_mask] = 0.0
                            Sk += W_eff.T @ Xb
                            Nk += W_eff.sum(axis=0)
                        else:
                            Sk += W.T @ Xb
                            Nk += wk_sum
                        denom = np.maximum(Nk[:, None], np.finfo(float).eps)
                        centers = Sk / denom
                    else:
                        lr = float(self.learning_rate)
                        for k in range(K):
                            if not update_mask[k]:
                                continue
                            wk = wk_sum[k]
                            if wk <= 0:
                                continue
                            xbar_k = (W[:, k][:, None] * Xb).sum(axis=0) / wk
                            centers[k] = (1.0 - lr) * centers[k] + lr * xbar_k

                    if self.learning_rate is not None and self.reassign_patience > 0:
                        to_reassign = np.where(empty_counts >= self.reassign_patience)[
                            0
                        ]
                        if to_reassign.size > 0:
                            far_idx = np.argmax(D2, axis=0)
                            for k in to_reassign:
                                centers[k] = Xb[far_idx[k]]
                                empty_counts[k] = 0

                center_shift_tot = np.linalg.norm(centers - prev, ord="fro")
                prev = centers.copy()
                obj_approx = self._approx_objective(X[monitor_idx], centers, alpha)
                obj_approx_hist.append(obj_approx)
                if self.verbose and (epoch % self.print_every == 0):
                    print(
                        f"[MiniBatchEKM] run {run+1}/{self.n_init} epoch {epoch}/{self.max_epochs}  center shift / var(X) = {center_shift_tot / Var_X:<.3e}  objective≈{obj_approx:<.3e}"
                    )
                if self.tol > 0.0 and center_shift_tot <= Var_X * self.tol:
                    if self.verbose:
                        print(
                            f"[MiniBatchEKM] Converged at epoch {epoch} (center shift / var(X) "
                            f"{center_shift_tot / Var_X:<.3e} <= tol {self.tol:<.3e})."
                        )
                    break

            # Evaluate full objective on X to select best run
            obj_full = self._approx_objective(X, centers, alpha)
            if obj_full < best_obj:
                best_obj = obj_full
                best_centers = centers.copy()
                best_alpha = float(alpha)
                best_epoch = int(epoch)
                best_counts = Nk.copy()
                best_sums = Sk.copy()
                best_obj_approx_hist = list(obj_approx_hist)

        # Assign best run results to estimator
        self.cluster_centers_ = best_centers
        self.alpha_ = best_alpha
        self.n_epochs_ = best_epoch
        self.counts_ = (
            best_counts if best_counts is not None else np.zeros(K, dtype=float)
        )
        self.sums_ = (
            best_sums
            if best_sums is not None
            else np.zeros((K, n_features), dtype=float)
        )
        self.objective_approx_ = (
            best_obj_approx_hist if best_obj_approx_hist is not None else []
        )

        D = _pairwise_distance(X, self.cluster_centers_, self.metric)
        D2 = D**2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        self.W_ = self._calc_weight(D2_shift, self.alpha_)
        E = np.exp(-self.alpha_ * D2_shift)
        self.U_ = E / np.sum(E, axis=1, keepdims=True)
        # Hard labels for training data (not maintained incrementally during partial_fit)
        self.labels_ = np.argmin(D, axis=1)
        return self

    def partial_fit(self, X_batch, y=None):
        """Incrementally update the model with a single mini-batch.

        Parameters
        ----------
        X_batch : array-like of shape (batch_size, n_features)
            Mini-batch of samples.
        y : Ignored
            For API consistency.

        Returns
        -------
        self : object
            Updated estimator.
        """
        # feature count consistency is enforced on subsequent calls.
        if not hasattr(self, "cluster_centers_") and not hasattr(
            self, "n_features_in_"
        ):
            Xb = validate_data(
                self,
                X_batch,
                accept_sparse=False,
                reset=True,
                dtype=[np.float64, np.float32],
                order="C",
                accept_large_sparse=False,
            )
        else:
            Xb = validate_data(
                self,
                X_batch,
                accept_sparse=False,
                reset=False,
                dtype=[np.float64, np.float32],
                order="C",
                accept_large_sparse=False,
            )
        rng = check_random_state(self.random_state)
        if not hasattr(self, "cluster_centers_"):
            self.cluster_centers_ = self._init_centers(Xb, rng)
            self.alpha_ = self._resolve_alpha(Xb, rng)
            K = self.n_clusters
            P = Xb.shape[1]
            self.counts_ = np.zeros(K, dtype=float)
            self.sums_ = np.zeros((K, P), dtype=float)
            self._empty_counts = np.zeros(K, dtype=np.int64)
        else:
            K = self.n_clusters
            if not hasattr(self, "_empty_counts"):
                self._empty_counts = np.zeros(K, dtype=np.int64)

        centers = self.cluster_centers_
        alpha = self.alpha_

        D2 = _pairwise_distance(Xb, centers, self.metric) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        W = self._calc_weight(D2_shift, alpha)
        wk_sum = W.sum(axis=0)
        update_mask = wk_sum > self.reassignment_ratio * Xb.shape[0]

        for k in range(K):
            if update_mask[k]:
                self._empty_counts[k] = 0
            else:
                self._empty_counts[k] += 1

        if self.learning_rate is None:
            if not np.all(update_mask):
                W_eff = W.copy()
                W_eff[:, ~update_mask] = 0.0
                self.sums_ += W_eff.T @ Xb
                self.counts_ += W_eff.sum(axis=0)
            else:
                self.sums_ += W.T @ Xb
                self.counts_ += wk_sum
            denom = np.maximum(self.counts_[:, None], np.finfo(float).eps)
            self.cluster_centers_ = self.sums_ / denom
        else:
            lr = float(self.learning_rate)
            for k in range(K):
                if not update_mask[k]:
                    continue
                wk = wk_sum[k]
                if wk <= 0:
                    continue
                xbar_k = (W[:, k][:, None] * Xb).sum(axis=0) / wk
                centers[k] = (1.0 - lr) * centers[k] + lr * xbar_k
            self.cluster_centers_ = centers

        if self.learning_rate is not None and self.reassign_patience > 0:
            to_reassign = np.where(self._empty_counts >= self.reassign_patience)[0]
            if to_reassign.size > 0:
                far_idx = np.argmax(D2, axis=0)
                for k in to_reassign:
                    self.cluster_centers_[k] = Xb[far_idx[k]]
                    self._empty_counts[k] = 0
        return self

    def predict(self, X):
        """Assign each sample in ``X`` to the closest learned center."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        D = _pairwise_distance(X, self.cluster_centers_, self.metric)
        labels = np.argmin(D, axis=1)
        distinct_clusters = len(set(labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )
        return labels

    def transform(self, X):
        """Compute distances from samples to cluster centers."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return _pairwise_distance(X, self.cluster_centers_, self.metric)

    def membership(self, X):
        """Compute soft membership matrix for samples in ``X``."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        alpha = self.alpha_
        D2 = self.transform(X) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        E = np.exp(-alpha * D2_shift)
        return E / (np.sum(E, axis=1, keepdims=True) + np.finfo(float).eps)

    def fit_predict(self, X, y=None):
        """Fit to ``X`` and return hard assignments."""
        return self.fit(X, y).predict(X)

    def fit_membership(self, X, y=None):
        """Fit to ``X`` and return the final membership matrix for training data."""
        return self.fit(X, y).U_
