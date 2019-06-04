
import numpy as np
from confound_prediction.deconfounding import DeConfounder


def test_deconfounder():
    rng = np.random.RandomState(0)

    # An in-sample test
    X = rng.normal(size=(100, 10))
    y = rng.normal(size=100)

    deconfounder = DeConfounder()
    deconfounder.fit(X, y)
    X_clean = deconfounder.transform(X, y)
    # Check that X_clean is indeed orthogonal to y
    np.testing.assert_almost_equal(X_clean.T.dot(y), 0)

    # An out-of-sample test

    # Generate data where X is a linear function of y
    y = rng.normal(size=100)
    coef = rng.normal(size=10)
    X = coef * y[:, np.newaxis]
    X_train = X[:-10]
    y_train = y[:-10]
    deconfounder.fit(X_train, y_train)
    X_clean = deconfounder.transform(X, y)
    # Check that X_clean is indeed orthogonal to y
    np.testing.assert_almost_equal(X_clean.T.dot(y), 0)
