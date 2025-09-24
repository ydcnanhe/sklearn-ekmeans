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
    ekm = EKMeans(n_clusters=2, random_state=0, n_init=2, max_iter=50)
    ekm.fit(X)
    assert ekm.cluster_centers_.shape == (2, 2)
    labels = ekm.predict(X)
    assert labels.shape[0] == X.shape[0]
    # membership
    U = ekm.membership(X)
    assert U.shape == (X.shape[0], 2)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-6)


def test_minibatchekm_basic():
    X = _toy_data()
    mb = MiniBatchEKMeans(n_clusters=2, random_state=0, max_epochs=5, batch_size=16)
    mb.fit(X)
    assert mb.cluster_centers_.shape == (2, 2)
    labels = mb.predict(X)
    assert labels.shape[0] == X.shape[0]
    U = mb.membership(X)
    assert U.shape == (X.shape[0], 2)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-6)
    # labels_ attribute should be present after full fit
    assert hasattr(mb, 'labels_')
    assert mb.labels_.shape == (X.shape[0],)


def test_alpha_dvariance():
    X = _toy_data()
    ekm = EKMeans(n_clusters=2, alpha='dvariance', random_state=0)
    ekm.fit(X)
    assert ekm.alpha_ > 0


def test_invalid_metric():
    X = _toy_data()
    with pytest.raises(ValueError):
        EKMeans(n_clusters=2, metric='cosine').fit(X)


def test_minibatch_partial_fit():
    X = _toy_data()
    mb = MiniBatchEKMeans(n_clusters=2, random_state=0, learning_rate=0.5)
    for i in range(0, X.shape[0], 10):
        mb.partial_fit(X[i:i+10])
    labels = mb.predict(X)
    assert labels.shape[0] == X.shape[0]
    # labels_ not maintained incrementally during partial_fit-only usage
    assert not hasattr(mb, 'labels_')

if __name__ == '__main__':
    test_ekm_estimator_checks()
    test_ekm_basic_fit_predict()
    test_minibatchekm_basic()
    test_alpha_dvariance()
    test_invalid_metric()
    test_minibatch_partial_fit()