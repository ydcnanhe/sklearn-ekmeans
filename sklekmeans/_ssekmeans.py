"""Semi-Supervised Equilibrium K-Means (SSEKM) estimators.

SSEKM extends EKMeans by incorporating semi-supervision through a prior
matrix of shape ``(n_samples, n_clusters)`` where each labeled row provides
per-class probabilities and an all-zero row indicates an unlabeled sample.
Pass this prior as ``prior_matrix=...`` to :meth:`fit` (or
``prior_matrix_batch=...`` to :meth:`partial_fit` for the mini-batch
variant). The weight update for labeled rows interpolates between the
equilibrium weight and the provided probabilities using a mixing parameter
``theta``.

Batch and mini-batch variants are provided: :class:`SSEKM` and
:class:`MiniBatchSSEKM`.

References
----------

.. [1] He, Y. (2025). Semi-supervised equilibrium K-means for imbalanced data clustering. Knowledge-Based Systems, 113990.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, _fit_context
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

# Optional numba acceleration (soft dependency)
try:  # pragma: no cover
    from numba import njit, prange, set_num_threads  # type: ignore

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False


def _pairwise_distance(X, Y=None, metric: str = "euclidean"):
    if metric == "euclidean":
        return euclidean_distances(X, Y, squared=False)
    if metric == "manhattan":
        return manhattan_distances(X, Y)
    raise ValueError(f"Unsupported distance metric: {metric!r}")


def _kmeans_plus_like(X, n_clusters, *, metric="euclidean", random_state=None):
    rng = check_random_state(random_state)
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=float)
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
    E = np.exp(-alpha * D2)
    U = E / np.sum(E, axis=1, keepdims=True)
    W = U * (1 - alpha * (D2 - np.sum(D2 * U, axis=1, keepdims=True)))
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    for i in zero_idx:
        pos = np.argmin(D2[i])
        W[i] = 0.0
        W[i, pos] = 1.0
    return W


if _NUMBA_AVAILABLE:  # pragma: no cover

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


class SSEKM(TransformerMixin, ClusterMixin, BaseEstimator):
    """Semi-Supervised Equilibrium K-Means (batch).

    Parameters
    ----------
    n_clusters : int, default=8
    metric : {'euclidean', 'manhattan'}, default='euclidean'
    alpha : float or {'dvariance'}, default='dvariance'
            Equilibrium weighting parameter (same as EKMeans). If ``'dvariance'``,
            a heuristic based on data variance is used.
    scale : float, default=2.0
                Multiplicative factor for the ``'dvariance'`` heuristic.
    theta : float or {'auto'}, default='auto'
            Supervision strength for labeled samples.

            - If a float is provided, it is used directly both in the supervised
                objective term and in the weight update for labeled rows:
                ``W = W_ekm + theta * b * (F_norm - W_ekm)``.
            - If ``'auto'``, set ``theta = |N| / |S|`` where ``|N|`` is the total
                number of samples and ``|S|`` is the number of labeled samples (rows
                of the prior with positive sum). When ``|S| = 0`` (no supervision),
                ``theta = 0`` and the estimator reduces to EKMeans.
    max_iter : int, default=300
    tol : float, default=1e-4
    n_init : int, default=1
    init : {'k-means', 'k-means++', 'random'} or ndarray, default='k-means++'
    random_state : int or None, default=None
    use_numba : bool, default=False
    numba_threads : int or None, default=None
    verbose : int, default=0

    Notes
    -----
    - Supervision is provided via a prior matrix ``F`` of shape
        ``(n_samples, n_clusters)``. Pass this as ``prior_matrix=F`` to
        :meth:`fit`. Rows with all zeros indicate unlabeled samples;
        otherwise values are class probabilities (labeled rows are
        row-normalized internally when positive).
    - Objective used for selection across initialisations is:
      ``sum(U * d^2) + theta * sum(b * (F - U) * d^2)`` where ``b`` is the
      labeled mask and ``U`` are exp-normalized memberships.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Hard assignment labels for training data.
    n_iter_ : int
        Number of iterations executed for the best initialisation.
    objective_ : float
        Objective value (including supervised term when provided) of the best run.
    alpha_ : float
        Resolved numeric alpha used during fitting.
    theta_super_ : float
        Resolved supervision strength used (``'auto'`` or numeric).
    W_ : ndarray of shape (n_samples, n_clusters)
        Final equilibrium weights; for labeled rows, incorporates the prior via
        the mixing with ``theta``.
    U_ : ndarray of shape (n_samples, n_clusters)
        Membership matrix based on exp-normalized distances before equilibrium
        correction. Each row sums to 1.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Methods
    -------
    fit(X, y=None, *, prior_matrix=None, F=None)
        Fit the model and learn cluster centers. ``prior_matrix`` supplies the
        supervision prior (``F`` kept as a backward-compatible alias).
    predict(X)
        Return the hard cluster label (nearest center) for each sample.
    transform(X)
        Return matrix of distances from samples to cluster centers.
    fit_predict(X, y=None, *, prior_matrix=None, F=None)
        Fit the model and return labels in one pass.
    membership(X)
        Compute soft membership (row-stochastic responsibilities).
    """

    _parameter_constraints = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions({"euclidean", "manhattan"})],
        "alpha": [Real, StrOptions({"dvariance"})],
        "scale": [Interval(Real, 0, None, closed="neither")],
        "theta": [Interval(Real, 0, None, closed="both"), StrOptions({"auto"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means", "k-means++", "random"}), np.ndarray],
        "random_state": [None, Integral],
        "use_numba": [bool],
        "numba_threads": [None, Interval(Integral, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        metric="euclidean",
        alpha="dvariance",
        scale=2.0,
        theta="auto",
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
        self.theta = theta
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self.use_numba = use_numba
        self.numba_threads = numba_threads
        self.verbose = verbose

    def _resolve_theta(self, F, n_samples):
        """Return (theta_super, theta_used) based on setting and labels.

        theta_super scales the supervised term in the objective and is also
        used directly in the labeled-row weight update when theta is 'auto'.
        For numeric theta, we use max(theta, 0) for both.
        """
        # numeric theta (allow >= 0 without upper bound)
        if not isinstance(self.theta, str):
            t = float(self.theta)
            if t < 0:
                t = 0.0
            return t
        # auto mode
        if F is None:
            return 0.0
        row_sum = np.sum(F, axis=1)
        S = int(np.sum(row_sum > 0))
        if S <= 0:
            return 0.0
        theta_super = float(n_samples) / float(S)
        # return pair for compatibility; blending uses theta_super directly
        return theta_super

    def _resolve_alpha(self, X):
        alpha = self.alpha
        if isinstance(alpha, str):
            if alpha == "dvariance":
                mu = np.mean(X, axis=0, keepdims=True)
                d2 = _pairwise_distance(X, mu, self.metric) ** 2
                dv = float(np.mean(d2))
                alpha = self.scale / max(dv, np.finfo(float).eps)
            else:  # pragma: no cover
                raise ValueError("Unsupported alpha option")
        return float(alpha)

    def _init_centers(self, X, rng):
        if isinstance(self.init, np.ndarray):
            centers = np.asarray(self.init, dtype=float)
            if centers.shape[0] != self.n_clusters:
                raise ValueError(
                    "init array should have shape (n_clusters, n_features)"
                )
            return centers.copy()
        if self.init == "k-means":
            km = KMeans(
                n_clusters=self.n_clusters,
                init="k-means++",
                n_init=1,
                max_iter=100,
                random_state=rng,
            )
            km.fit(X)
            centers = km.cluster_centers_.astype(float, copy=False)
        elif self.init == "k-means++":
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

    def _calc_weight(self, D2, alpha):
        if self.use_numba and _NUMBA_AVAILABLE:
            if self.numba_threads is not None:
                try:
                    set_num_threads(int(self.numba_threads))  # type: ignore
                except Exception:  # pragma: no cover
                    pass
            return _calc_weight_numba(D2, alpha)  # type: ignore
        return _calc_weight_numpy(D2, alpha)

    def _objective(self, X, centers, alpha, F=None, theta_super=None):
        D2 = _pairwise_distance(X, centers, self.metric) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        E = np.exp(-alpha * D2_shift)
        U = E / np.sum(E, axis=1, keepdims=True)
        obj = float(np.sum(U * D2))
        if F is not None:
            b = (np.sum(F, axis=1) > 0).astype(float)[:, None]
            # normalize labeled rows defensively
            row_sum = np.sum(F, axis=1, keepdims=True)
            F_norm = np.divide(F, row_sum, out=np.zeros_like(F), where=row_sum > 0)
            if theta_super is None:
                # fallback to numeric theta if provided
                theta_super = (
                    float(self.theta) if not isinstance(self.theta, str) else 0.0
                )
            obj += float(theta_super * np.sum(b * (F_norm - U) * D2))
        return obj

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, *, prior_matrix=None, F=None):
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=True,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        # Allow both names temporarily; prefer prior_matrix
        if prior_matrix is not None and F is not None:
            raise ValueError("Provide either prior_matrix or F, not both.")
        F = prior_matrix if prior_matrix is not None else F
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        K = self.n_clusters
        alpha = self._resolve_alpha(X)
        verbose = self.verbose
        Var_X = np.mean(np.var(X, axis=0))

        if F is not None:
            F = np.asarray(F, dtype=float)
            if F.shape != (n_samples, K):
                raise ValueError(
                    f"F must have shape (n_samples, n_clusters) = {(n_samples, K)}, got {F.shape}"
                )
        # resolve theta for objective scaling and mixing
        theta_super = self._resolve_theta(F, n_samples)

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
                W_ekm = self._calc_weight(D2_shift, alpha)
                if F is not None:
                    b = (np.sum(F, axis=1) > 0).astype(float)[:, None]
                    row_sum = np.sum(F, axis=1, keepdims=True)
                    F_norm = np.divide(
                        F, row_sum, out=np.zeros_like(F), where=row_sum > 0
                    )
                    W = W_ekm + theta_super * b * (F_norm - W_ekm)
                    # The above yields W_ekm for unlabeled rows, and convex blend for labeled rows
                else:
                    W = W_ekm

                for k in range(K):
                    sw = np.sum(W[:, k])
                    centers[k] = (W[:, k] @ X) / (sw + np.finfo(float).eps)

                if verbose:
                    obj_it = self._objective(X, centers, alpha, F, theta_super)
                    print(f"Iteration {it}, objective/loss {obj_it:<.3e}.")

                center_shift_tot = np.linalg.norm(centers - prev, "fro")
                if self.tol > 0.0 and center_shift_tot <= Var_X * self.tol:
                    if verbose:
                        print(
                            f"Converged at iteration {it} (center shift / var(X) "
                            f"{center_shift_tot / Var_X:<.3e} <= tol {self.tol:<.3e})."
                        )
                    break
                prev = centers.copy()

            if (
                verbose
                and it == self.max_iter - 1
                and center_shift_tot > Var_X * self.tol
            ):
                print(
                    f"Reached max_iter {self.max_iter} (center shift / var(X) "
                    f"{center_shift_tot / Var_X:<.3e} > tol {self.tol:<.3e})."
                )

            D2_eval = _pairwise_distance(X, centers, self.metric) ** 2
            obj = self._objective(X, centers, alpha, F, theta_super)
            if obj < best_obj:
                best_obj = obj
                best_centers = centers
                best_labels = np.argmin(D2_eval, axis=1)
                best_niter = it

        distinct = len(set(best_labels))
        if distinct < self.n_clusters:
            warnings.warn(
                f"Number of distinct clusters ({distinct}) found smaller than n_clusters ({self.n_clusters}). Possibly due to duplicate points in X.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.n_iter_ = best_niter
        self.objective_ = float(best_obj)
        self.alpha_ = alpha

        D = _pairwise_distance(X, self.cluster_centers_, self.metric)
        D2 = D**2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        W_ekm = self._calc_weight(D2_shift, alpha)
        if F is not None:
            b = (np.sum(F, axis=1) > 0).astype(float)[:, None]
            row_sum = np.sum(F, axis=1, keepdims=True)
            F_norm = np.divide(F, row_sum, out=np.zeros_like(F), where=row_sum > 0)
            self.W_ = W_ekm + theta_super * b * (F_norm - W_ekm)
        else:
            self.W_ = W_ekm
        E = np.exp(-alpha * D2_shift)
        self.U_ = E / np.sum(E, axis=1, keepdims=True)
        # store resolved theta values for reference
        self.theta_super_ = float(theta_super)
        return self

    def predict(self, X):
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
        return np.argmin(D, axis=1)

    def transform(self, X):
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

    def fit_predict(self, X, y=None, *, prior_matrix=None, F=None):
        if prior_matrix is not None and F is not None:
            raise ValueError("Provide either prior_matrix or F, not both.")
        return self.fit(X, y=y, prior_matrix=prior_matrix, F=F).labels_

    def membership(self, X):
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
        return E / np.sum(E, axis=1, keepdims=True)


class MiniBatchSSEKM(TransformerMixin, ClusterMixin, BaseEstimator):
    """Mini-batch SSEKM.

    Mini-batch optimisation of the semi-supervised equilibrium k-means
    objective. Supervision is provided via a prior matrix, using the
    ``prior_matrix`` keyword to :meth:`fit` and ``prior_matrix_batch`` to
    :meth:`partial_fit`. Labeled rows in the prior influence weights via the
    mixing factor ``theta``.

    Parameters
    ----------
    n_clusters : int, default=8
    metric : {'euclidean', 'manhattan'}, default='euclidean'
    alpha : float or {'dvariance'}, default='dvariance'
        Equilibrium weighting parameter (``'dvariance'`` uses a subsample to
        estimate a heuristic value scaled by ``scale``).
    scale : float, default=2.0
        Scaling factor for the heuristic alpha.
    theta : float or {'auto'}, default='auto'
        Supervision strength. ``'auto'`` sets ``theta = |N| / |S|``. Numeric
        values are used directly in both the objective and the labeled-row
        weight update.
    batch_size : int, default=256
    max_epochs : int, default=10
    n_init : int, default=1
    init : {'k-means', 'k-means++', 'random'} or ndarray, default='k-means++'
    init_size : int or None, default=None
    shuffle : bool, default=True
    learning_rate : float or None, default=None
    tol : float, default=1e-4
    reassignment_ratio : float, default=0.0
    reassign_patience : int, default=3
    verbose : int, default=0
    monitor_size : int or None, default=1024
    print_every : int, default=1
    use_numba : bool, default=False
    numba_threads : int or None, default=None
    random_state : int or None, default=None

    Notes
    -----
    - Provide the full-dataset prior using ``prior_matrix`` to :meth:`fit`,
      or mini-batch priors using ``prior_matrix_batch`` to
      :meth:`partial_fit`.
    - Unlabeled rows are all zeros; labeled rows are row-normalized when
      positive.
    - The monitoring objective returned in ``objective_approx_`` includes the
      supervised term scaled by ``theta`` when a prior is provided.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final centers after training.
    labels_ : ndarray of shape (n_samples,)
        Hard assignment labels for the training data (available after :meth:`fit`).
    alpha_ : float
        Resolved alpha value.
    theta_super_ : float
        Resolved supervision strength used (``'auto'`` or numeric).
    objective_approx_ : list of float
        Epoch-wise approximate objectives measured on a monitoring subset.
    counts_ : ndarray of shape (n_clusters,)
        Accumulated batch weights per cluster (accumulation mode; present after :meth:`fit`).
    sums_ : ndarray of shape (n_clusters, n_features)
        Accumulated weighted sums per cluster (accumulation mode; present after :meth:`fit`).
    W_, U_ : ndarrays
        Final equilibrium weights and memberships for the full training data (set by :meth:`fit`).
    n_epochs_ : int
        Number of epochs run in the best initialisation.
    n_features_in_ : int
        Number of features seen during the first call to :meth:`fit` or :meth:`partial_fit`.

    Methods
    -------
    fit(X, y=None, *, prior_matrix=None, F=None)
        Run full mini-batch training with optional prior supervision.
    partial_fit(X_batch, y=None, *, prior_matrix_batch=None, F_batch=None)
        Update model parameters using a single mini-batch and optional batch prior.
    predict(X)
        Return hard cluster labels for samples.
    transform(X)
        Return distances from samples to cluster centers.
    fit_predict(X, y=None, *, prior_matrix=None, F=None)
        Fit the model and return labels for X.
    membership(X)
        Compute soft membership for input samples.
    """

    _parameter_constraints = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions({"euclidean", "manhattan"})],
        "alpha": [Real, StrOptions({"dvariance"})],
        "scale": [Interval(Real, 0, None, closed="neither")],
        "theta": [Interval(Real, 0, None, closed="both"), StrOptions({"auto"})],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "max_epochs": [Interval(Integral, 1, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means", "k-means++", "random"}), np.ndarray],
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
        alpha="dvariance",
        scale=2.0,
        theta="auto",
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
        self.theta = theta
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

    def _resolve_theta(self, F, n_samples):
        # numeric theta in [0,1]
        if not isinstance(self.theta, str):
            t = float(self.theta)
            if t < 0:
                t = 0.0
            return t
        if F is None:
            return 0.0
        row_sum = np.sum(F, axis=1)
        S = int(np.sum(row_sum > 0))
        if S <= 0:
            return 0.0
        theta_super = float(n_samples) / float(S)
        return theta_super

    def _init_centers(self, X, rng):
        if isinstance(self.init, np.ndarray):
            return np.asarray(self.init, dtype=float).copy()
        if self.init == "k-means":
            km = KMeans(
                n_clusters=self.n_clusters,
                init="k-means++",
                n_init=1,
                max_iter=100,
                random_state=rng,
            )
            km.fit(X)
            centers = km.cluster_centers_.astype(float, copy=False)
        elif self.init == "k-means++":
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
        alpha = self.alpha
        if isinstance(alpha, str):
            if alpha == "dvariance":
                if self.init_size is None:
                    n0 = min(X.shape[0], max(10 * self.n_clusters, self.batch_size))
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
        if self.use_numba and _NUMBA_AVAILABLE:
            if self.numba_threads is not None:
                try:
                    set_num_threads(int(self.numba_threads))  # type: ignore
                except Exception:  # pragma: no cover
                    pass
            return _calc_weight_numba(D2, alpha)  # type: ignore
        return _calc_weight_numpy(D2, alpha)

    def _approx_objective(self, Xs, centers, alpha, Fs=None, theta_super=None):
        D2 = _pairwise_distance(Xs, centers, self.metric) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        E = np.exp(-alpha * D2_shift)
        U = E / np.sum(E, axis=1, keepdims=True)
        obj = float(np.sum(U * D2))
        if Fs is not None:
            b = (np.sum(Fs, axis=1) > 0).astype(float)[:, None]
            row_sum = np.sum(Fs, axis=1, keepdims=True)
            F_norm = np.divide(Fs, row_sum, out=np.zeros_like(Fs), where=row_sum > 0)
            if theta_super is None:
                theta_super = (
                    float(self.theta) if not isinstance(self.theta, str) else 0.0
                )
            obj += float(theta_super * np.sum(b * (F_norm - U) * D2))
        return obj

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, *, prior_matrix=None, F=None):
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=True,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        # Allow both names temporarily; prefer prior_matrix
        if prior_matrix is not None and F is not None:
            raise ValueError("Provide either prior_matrix or F, not both.")
        F = prior_matrix if prior_matrix is not None else F
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        K = self.n_clusters
        Var_X = np.mean(np.var(X, axis=0))

        if F is not None:
            F = np.asarray(F, dtype=float)
            if F.shape != (n_samples, K):
                raise ValueError(
                    f"F must have shape (n_samples, n_clusters) = {(n_samples, K)}, got {F.shape}"
                )
        theta_super = self._resolve_theta(F, n_samples)

        best_obj = np.inf
        best_centers = None
        best_alpha = None
        best_epoch = None
        best_counts = None
        best_sums = None
        best_obj_hist = None

        for run in range(self.n_init):
            centers = self._init_centers(X, rng)
            alpha = self._resolve_alpha(X, rng)

            prev = centers.copy()
            empty_counts = np.zeros(K, dtype=np.int64)
            obj_hist = []

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
                    Fb = F[batch_idx] if F is not None else None

                    D2 = _pairwise_distance(Xb, centers, self.metric) ** 2
                    D2_shift = D2 - D2.min(axis=1, keepdims=True)
                    W_ekm = self._calc_weight(D2_shift, alpha)

                    if Fb is not None:
                        b = (np.sum(Fb, axis=1) > 0).astype(float)[:, None]
                        row_sum = np.sum(Fb, axis=1, keepdims=True)
                        F_norm = np.divide(
                            Fb, row_sum, out=np.zeros_like(Fb), where=row_sum > 0
                        )
                        W = W_ekm + theta_super * b * (F_norm - W_ekm)
                    else:
                        W = W_ekm

                    wk_sum = W.sum(axis=0)
                    update_mask = np.abs(wk_sum) > self.reassignment_ratio * Xb.shape[0]

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
                        denom = Nk[:, None]
                        centers = Sk / (denom + np.finfo(float).eps)
                    else:
                        lr = float(self.learning_rate)
                        for k in range(K):
                            if not update_mask[k]:
                                continue
                            wk = wk_sum[k]
                            xbar_k = (W[:, k][:, None] * Xb).sum(axis=0) / (
                                wk + np.finfo(float).eps
                            )
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
                Fs = F[monitor_idx] if F is not None else None
                obj_approx = self._approx_objective(
                    X[monitor_idx], centers, alpha, Fs, theta_super
                )
                obj_hist.append(obj_approx)
                if self.verbose and (epoch % self.print_every == 0):
                    print(
                        f"[MiniBatchSSEKM] run {run+1}/{self.n_init} epoch {epoch}/{self.max_epochs}  center shift / var(X) = {center_shift_tot / Var_X:<.3e}  objective≈{obj_approx:<.3e}"
                    )
                if self.tol > 0.0 and center_shift_tot <= Var_X * self.tol:
                    if self.verbose:
                        print(
                            f"[MiniBatchSSEKM] Converged at epoch {epoch} (center shift / var(X) "
                            f"{center_shift_tot / Var_X:<.3e} <= tol {self.tol:<.3e})."
                        )
                    break

            obj_full = self._approx_objective(X, centers, alpha, F, theta_super)
            if obj_full < best_obj:
                best_obj = obj_full
                best_centers = centers.copy()
                best_alpha = float(alpha)
                best_epoch = int(epoch)
                best_counts = Nk.copy()
                best_sums = Sk.copy()
                best_obj_hist = list(obj_hist)

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
        self.objective_approx_ = best_obj_hist if best_obj_hist is not None else []

        D = _pairwise_distance(X, self.cluster_centers_, self.metric)
        D2 = D**2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        W_ekm = self._calc_weight(D2_shift, self.alpha_)
        if F is not None:
            b = (np.sum(F, axis=1) > 0).astype(float)[:, None]
            row_sum = np.sum(F, axis=1, keepdims=True)
            F_norm = np.divide(F, row_sum, out=np.zeros_like(F), where=row_sum > 0)
            self.W_ = W_ekm + theta_super * b * (F_norm - W_ekm)
        else:
            self.W_ = W_ekm
        E = np.exp(-self.alpha_ * D2_shift)
        self.U_ = E / np.sum(E, axis=1, keepdims=True)
        self.labels_ = np.argmin(D, axis=1)
        self.theta_super_ = float(theta_super)
        return self

    def partial_fit(self, X_batch, y=None, *, prior_matrix_batch=None, F_batch=None):
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
        # Allow both names temporarily; prefer prior_matrix_batch
        if prior_matrix_batch is not None and F_batch is not None:
            raise ValueError("Provide either prior_matrix_batch or F_batch, not both.")
        F_batch = prior_matrix_batch if prior_matrix_batch is not None else F_batch
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
        W_ekm = self._calc_weight(D2_shift, alpha)
        if F_batch is not None:
            Fb = np.asarray(F_batch, dtype=float)
            if Fb.shape[0] != Xb.shape[0] or Fb.shape[1] != K:
                raise ValueError("F_batch must have shape (batch_size, n_clusters)")
            b = (np.sum(Fb, axis=1) > 0).astype(float)[:, None]
            row_sum = np.sum(Fb, axis=1, keepdims=True)
            F_norm = np.divide(Fb, row_sum, out=np.zeros_like(Fb), where=row_sum > 0)
            # resolve theta using stored values if available; otherwise batch-based
            if hasattr(self, "theta_super_"):
                theta_used = float(self.theta_super_)
            elif isinstance(self.theta, str):
                Sb = float(np.sum(row_sum > 0))
                theta_used = 0.0 if Sb <= 0 else (Xb.shape[0] / Sb)
            else:
                theta_used = float(self.theta)
                # keep numeric theta as provided; do not bound to [0,1]
            W = W_ekm + theta_used * b * (F_norm - W_ekm)
        else:
            W = W_ekm

        wk_sum = W.sum(axis=0)
        update_mask = np.abs(wk_sum) > self.reassignment_ratio * Xb.shape[0]

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
            denom = self.counts_[:, None]
            self.cluster_centers_ = self.sums_ / (denom + np.finfo(float).eps)
        else:
            lr = float(self.learning_rate)
            for k in range(K):
                if not update_mask[k]:
                    continue
                wk = wk_sum[k]
                xbar_k = (W[:, k][:, None] * Xb).sum(axis=0) / (
                    wk + np.finfo(float).eps
                )
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

    def fit_predict(self, X, y=None, *, prior_matrix=None, F=None):
        if prior_matrix is not None and F is not None:
            raise ValueError("Provide either prior_matrix or F, not both.")
        return self.fit(X, y=y, prior_matrix=prior_matrix, F=F).labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to."""
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
        return np.argmin(D, axis=1)

    def transform(self, X):
        """Transform X to a cluster-distance space (pairwise distances)."""
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
        """Soft membership (U) computed from distances using current alpha_."""
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
        return E / np.sum(E, axis=1, keepdims=True)
