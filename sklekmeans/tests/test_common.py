"""Common estimator tests for sklekmeans using scikit-learn's parametrize_with_checks."""
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklekmeans.utils import all_estimators

@parametrize_with_checks([est() for _, est in all_estimators()])
def test_estimators(estimator, check, request):
    check(estimator)
