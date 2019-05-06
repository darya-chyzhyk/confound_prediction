import numpy as np

from confound_isolating.mutual_information import (_entropy,
                                                   _entropy_gaussian,
                                                   _nearest_distances,
                                                   mutual_information)



def test_entropy():
    # Testing against correlated Gaussian variables
    # (analytical results are known)
    # Entropy of a 3-dimensional gaussian variable
    rng = np.random.RandomState(0)
    n = 50000
    d = 3
    P = np.array([[1, 0, 0], [0, 1, .5], [0, 0, 1]])
    C = np.dot(P, P.T)
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    H_th = _entropy_gaussian(C)
    H_est = _entropy(X.T, k=5)
    # Our estimated entropy should always be less that the actual one
    # (entropy estimation undershoots) but not too much
    np.testing.assert_array_less(H_est, H_th)
    np.testing.assert_array_less(.9*H_th, H_est)


def test_mutual_information():
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    #P = np.random.randn(2, 2)
    P = np.array([[1, 0], [0.5, 1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = mutual_information((X, Y), k=5)
    MI_th = (entropy_gaussian(C[0, 0])
             + entropy_gaussian(C[1, 1])
             - entropy_gaussian(C)
            )
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    print(MI_est, MI_th)
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .3)


def test_degenerate():
    # Test that our estimators are well-behaved with regards to
    # degenerate solutions
    rng = np.random.RandomState(0)
    x = rng.randn(50000)
    X = np.c_[x, x]
    assert np.isfinite(entropy(X))
    assert np.isfinite(mutual_information((x[:, np.newaxis],
                                           x[:,  np.newaxis])))
    assert 2.9 < mutual_information_2d(x, x) < 3.1


def test_mutual_information_2d():
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    #P = np.random.randn(2, 2)
    P = np.array([[1, 0], [.9, .1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = mutual_information_2d(X.ravel(), Y.ravel())
    MI_th = (entropy_gaussian(C[0, 0])
             + entropy_gaussian(C[1, 1])
             - entropy_gaussian(C)
            )
    print(MI_est, MI_th)
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .2)


if __name__ == '__main__':
    # Run our tests
    test_entropy()
    test_mutual_information()
    test_degenerate()
    test_mutual_information_2d()
