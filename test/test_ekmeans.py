import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from sklekmeans import EKMeans, MiniBatchEKMeans


def test_ekm_estimator_checks():
    check_estimator(EKMeans())
    check_estimator(MiniBatchEKMeans())


def _toy_data():
    rng = np.random.RandomState(0)
    X1 = rng.normal(loc=0.0, scale=0.3, size=(30, 2))
    X2 = rng.normal(loc=5.0, scale=0.3, size=(10, 2))  # imbalance
    return np.vstack([X1, X2])


def test_ekm_basic_fit_predict():
    X = _toy_data()
    ekm = EKMeans(n_clusters=2, random_state=0, n_init=2, max_iter=50, verbose=1)
    ekm.fit(X)
    assert ekm.cluster_centers_.shape == (2, 2)
    labels = ekm.predict(X)
    assert labels.shape[0] == X.shape[0]
    # membership
    U = ekm.membership(X)
    assert U.shape == (X.shape[0], 2)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-6)


def test_minibatchekm_acc_basic():
    X = _toy_data()
    mb = MiniBatchEKMeans(
        n_clusters=2, random_state=0, n_init=2, max_epochs=5, batch_size=16, verbose=1
    )
    mb.fit(X)
    assert mb.cluster_centers_.shape == (2, 2)
    labels = mb.predict(X)
    assert labels.shape[0] == X.shape[0]
    U = mb.membership(X)
    assert U.shape == (X.shape[0], 2)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-6)
    # labels_ attribute should be present after full fit
    assert hasattr(mb, "labels_")
    assert mb.labels_.shape == (X.shape[0],)


def test_minibatchekm_online_basic():
    X = _toy_data()
    mb = MiniBatchEKMeans(
        n_clusters=2,
        random_state=0,
        n_init=2,
        max_epochs=5,
        batch_size=16,
        learning_rate=0.1,
        verbose=1,
    )
    mb.fit(X)
    assert mb.cluster_centers_.shape == (2, 2)


def test_fit_membership():
    X = _toy_data()
    ekm_nb = EKMeans(
        n_clusters=2,
        random_state=0,
        n_init=2,
        max_iter=50,
        use_numba=True,
        numba_threads=1,
    )
    mbekm_nb = MiniBatchEKMeans(
        n_clusters=2,
        random_state=0,
        n_init=2,
        max_epochs=5,
        batch_size=16,
        use_numba=True,
        numba_threads=1,
    )
    U_ekm = ekm_nb.fit_membership(X)
    U_mbekm = mbekm_nb.fit_membership(X)
    assert U_ekm.shape == (X.shape[0], 2)
    assert U_mbekm.shape == (X.shape[0], 2)


def test_alpha_dvariance():
    X = _toy_data()
    ekm = EKMeans(n_clusters=2, alpha="dvariance", random_state=0)
    minibatchekm = MiniBatchEKMeans(n_clusters=2, alpha="dvariance", random_state=0)
    ekm.fit(X)
    minibatchekm.fit(X)
    assert ekm.alpha_ > 0
    assert minibatchekm.alpha_ > 0


def test_invalid_metric():
    X = _toy_data()
    with pytest.raises(ValueError):
        EKMeans(n_clusters=2, metric="cosine").fit(X)


def test_minibatch_partial_fit():
    X = _toy_data()
    mb = MiniBatchEKMeans(n_clusters=2, random_state=0, learning_rate=0.5)
    for i in range(0, X.shape[0], 10):
        mb.partial_fit(X[i : i + 10])
    labels = mb.predict(X)
    assert labels.shape[0] == X.shape[0]
    # labels_ not maintained incrementally during partial_fit-only usage
    assert not hasattr(mb, "labels_")


def test_different_metrics_and_init_array():
    X = _toy_data()
    # init from small subset
    init = X[:2].copy()
    ekm_eu = EKMeans(n_clusters=2, metric="euclidean", random_state=0, init=init).fit(X)
    ekm_ma = EKMeans(n_clusters=2, metric="manhattan", random_state=0, n_init=5).fit(X)
    ekm_rd = EKMeans(
        n_clusters=2, metric="euclidean", random_state=0, init="random"
    ).fit(X)
    mbekm_rd = MiniBatchEKMeans(
        n_clusters=2, metric="euclidean", random_state=0, init="random"
    ).fit(X)
    assert (
        ekm_eu.cluster_centers_.shape
        == ekm_ma.cluster_centers_.shape
        == ekm_rd.cluster_centers_.shape
        == mbekm_rd.cluster_centers_.shape
        == (2, X.shape[1])
    )


if __name__ == "__main__":
    test_ekm_estimator_checks()
    test_ekm_basic_fit_predict()
    test_minibatchekm_acc_basic()
    test_minibatchekm_online_basic()
    test_fit_membership()
    test_alpha_dvariance()
    test_invalid_metric()
    test_minibatch_partial_fit()
    test_different_metrics_and_init_array()
