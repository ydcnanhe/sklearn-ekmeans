import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from sklekmeans import SSEKM, EKMeans, MiniBatchSSEKM


def test_ssekm_estimator_checks():
    check_estimator(SSEKM())
    check_estimator(MiniBatchSSEKM())


def _toy_data(seed=0):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(loc=0.0, scale=0.3, size=(30, 2))
    X2 = rng.normal(loc=5.0, scale=0.3, size=(10, 2))  # imbalance
    return np.vstack([X1, X2])


def test_ssekmeans_unlabeled_equivalence_to_ekm():
    X = _toy_data()
    K = 2
    F = np.zeros((X.shape[0], K), dtype=float)

    ekm = EKMeans(n_clusters=K, random_state=0, n_init=2, max_iter=50, alpha=0.5)
    ssekm = SSEKM(
        n_clusters=K,
        random_state=0,
        n_init=2,
        max_iter=50,
        alpha=0.5,
        theta=0.7,  # should be ignored for unlabeled rows
    )

    ekm.fit(X)
    ssekm.fit(X, F=F)

    # For unlabeled F, SSEKM weights should match EKMeans weights
    assert np.allclose(ssekm.W_, ekm.W_, atol=1e-6)
    # Centers should also be very close
    assert np.allclose(ssekm.cluster_centers_, ekm.cluster_centers_, atol=1e-5)


def test_ssekmeans_supervised_rows_follow_F_onehot_theta1():
    X = _toy_data()
    K = 2
    # Create one-hot labels for a subset of rows
    F = np.zeros((X.shape[0], K), dtype=float)
    labeled_idx = np.arange(0, 10)  # label first 10 samples as class 0
    F[labeled_idx, 0] = 1.0

    ssekm = SSEKM(
        n_clusters=K,
        random_state=0,
        n_init=1,
        max_iter=50,
        alpha=0.5,
        theta=1.0,  # fully trust provided F on labeled rows
    )
    ssekm.fit(X, F=F)

    # On labeled rows with one-hot F and theta=1, W_ should equal F exactly
    assert np.allclose(ssekm.W_[labeled_idx], F[labeled_idx], atol=1e-8)
    # Each row still sums to 1
    assert np.allclose(ssekm.W_.sum(axis=1), 1.0, atol=1e-6)


def test_ssekmeans_theta_zero_matches_ekm():
    X = _toy_data()
    K = 2
    # Provide some labels but set theta=0 so supervision is ignored
    F = np.zeros((X.shape[0], K), dtype=float)
    F[:8, 0] = 1.0

    ekm = EKMeans(n_clusters=K, random_state=0, n_init=2, max_iter=50, alpha=0.5)
    ssekm = SSEKM(
        n_clusters=K,
        random_state=0,
        n_init=2,
        max_iter=50,
        alpha=0.5,
        theta=0.0,  # ignore supervision
    )

    ekm.fit(X)
    ssekm.fit(X, F=F)

    assert np.allclose(ssekm.W_, ekm.W_, atol=1e-6)
    assert np.allclose(ssekm.cluster_centers_, ekm.cluster_centers_, atol=1e-5)


def test_minibatch_ssekmeans_basic_shapes():
    X = _toy_data()
    K = 2
    F = np.zeros((X.shape[0], K), dtype=float)
    F[:5, 0] = 1.0

    mb = MiniBatchSSEKM(
        n_clusters=K,
        random_state=0,
        n_init=1,
        max_epochs=5,
        batch_size=16,
        alpha=0.5,
        theta=0.5,
        verbose=1,
    )
    mb.fit(X, F=F)
    assert mb.cluster_centers_.shape == (K, X.shape[1])
    assert hasattr(mb, "U_") and hasattr(mb, "W_")
    assert np.allclose(mb.U_.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    test_ssekm_estimator_checks()
    test_ssekmeans_unlabeled_equivalence_to_ekm()
    test_ssekmeans_supervised_rows_follow_F_onehot_theta1()
    test_ssekmeans_theta_zero_matches_ekm()
    test_minibatch_ssekmeans_basic_shapes()
