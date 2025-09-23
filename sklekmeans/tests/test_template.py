"""Tests for sklekmeans template estimators."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal

from sklekmeans import TemplateClassifier, TemplateEstimator, TemplateTransformer


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_estimator(data):
    est = TemplateEstimator()
    assert est.demo_param == "demo_param"
    est.fit(*data)
    assert hasattr(est, "is_fitted_")
    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_template_transformer(data):
    X, y = data
    trans = TemplateTransformer()
    assert trans.demo_param == "demo"
    trans.fit(X)
    assert trans.n_features_in_ == X.shape[1]
    X_trans = trans.transform(X)
    assert_allclose(X_trans, np.sqrt(X))
    X_trans2 = trans.fit_transform(X)
    assert_allclose(X_trans2, np.sqrt(X))


def test_template_classifier(data):
    X, y = data
    clf = TemplateClassifier()
    assert clf.demo_param == "demo"
    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
