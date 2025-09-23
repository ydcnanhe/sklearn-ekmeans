"""Template estimators (migrated from original project template)."""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class TemplateEstimator(BaseEstimator):
	"""A template estimator to be used as a reference implementation.

	For more information regarding how to build your own estimator, read more
	in the :ref:`User Guide <user_guide>`.

	Parameters
	----------
	demo_param : str, default='demo_param'
		A parameter used for demonstration of how to pass and store parameters.

	Attributes
	----------
	is_fitted_ : bool
		A boolean indicating whether the estimator has been fitted.

	n_features_in_ : int
		Number of features seen during :term:`fit`.

	feature_names_in_ : ndarray of shape (`n_features_in_`,)
		Names of features seen during :term:`fit`. Defined only when `X`
		has feature names that are all strings.

	Examples
	--------
	>>> from sklekmeans import TemplateEstimator
	>>> import numpy as np
	>>> X = np.arange(100).reshape(100, 1)
	>>> y = np.zeros((100, ))
	>>> estimator = TemplateEstimator()
	>>> estimator.fit(X, y)
	TemplateEstimator()
	"""

	_parameter_constraints = {
		"demo_param": [str],
	}

	def __init__(self, demo_param="demo_param"):
		self.demo_param = demo_param

	@_fit_context(prefer_skip_nested_validation=True)
	def fit(self, X, y):
		X, y = self._validate_data(X, y, accept_sparse=True)
		self.is_fitted_ = True
		return self

	def predict(self, X):
		check_is_fitted(self)
		X = self._validate_data(X, accept_sparse=True, reset=False)
		return np.ones(X.shape[0], dtype=np.int64)


class TemplateClassifier(ClassifierMixin, BaseEstimator):
	"""An example classifier which implements a 1-NN algorithm.

	Examples
	--------
	>>> from sklearn.datasets import load_iris
	>>> from sklekmeans import TemplateClassifier
	>>> X, y = load_iris(return_X_y=True)
	>>> clf = TemplateClassifier().fit(X, y)
	>>> clf.predict(X).shape
	(150,)
	"""

	_parameter_constraints = {
		"demo_param": [str],
	}

	def __init__(self, demo_param="demo"):
		self.demo_param = demo_param

	@_fit_context(prefer_skip_nested_validation=True)
	def fit(self, X, y):
		X, y = self._validate_data(X, y)
		check_classification_targets(y)
		self.classes_ = np.unique(y)
		self.X_ = X
		self.y_ = y
		return self

	def predict(self, X):
		check_is_fitted(self)
		X = self._validate_data(X, reset=False)
		dist = euclidean_distances(X, self.X_)
		inds = np.argmin(dist, axis=1)
		return self.y_[inds]


class TemplateTransformer(TransformerMixin, BaseEstimator):
	"""An example transformer that returns the element-wise square root.

	Parameters
	----------
	demo_param : str, default='demo'
		A parameter used for demonstration.
	"""

	_parameter_constraints = {
		"demo_param": [str],
	}

	def __init__(self, demo_param="demo"):
		self.demo_param = demo_param

	@_fit_context(prefer_skip_nested_validation=True)
	def fit(self, X, y=None):
		X = self._validate_data(X, accept_sparse=True)
		return self

	def transform(self, X):
		check_is_fitted(self, attributes=["n_features_in_"])
		X = self._validate_data(X, accept_sparse=True, reset=False)
		return np.sqrt(X)
