import numpy as np

from confound_isolating.mutual_information import (_entropy,
                                                   _entropy_gaussian,
                                                   mutual_information,
                                                   mutual_kde)



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
    MI_th = (_entropy_gaussian(C[0, 0])
             + _entropy_gaussian(C[1, 1])
             - _entropy_gaussian(C)
            )
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    print(MI_est, MI_th)
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .3)


def test_mutual_kde():
    # Test if mutual information is non-negative
    # Test if mutual information for the correlated variables is bigger than
    # for uncorrelated

    xx = np.array([-1,1])
    yy = np.array([0, 10])
    means = [xx.mean(), yy.mean()]
    stds = [xx.std() / 3, yy.std() / 3]

    # Uncorrelated values
    corr = 0.01  # correlation
    covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
            [stds[0] * stds[1] * corr, stds[1] ** 2]]

    np.random.seed(42)
    x_uncorr, y_uncorr = np.random.multivariate_normal(means, covs, 1000).T
    mutual_uncorr = mutual_kde(x_uncorr, y_uncorr, type_bandwidth='scott')

    #print('Mutual information uncorrelated variables: ',

    # Correlated values
    # TODO test on the corr = 0.9, kde_xy_values contains nan and mutual
    #  information return the error

    corr = 0.8  # correlation
    covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
            [stds[0] * stds[1] * corr, stds[1] ** 2]]

    # x, y = np.random.multivariate_normal(means, covs, 1000).T
    np.random.seed(42)
    x_corr, y_corr = np.random.multivariate_normal(means, covs, 1000).T
    mutual_corr = mutual_kde(x_corr, y_corr, type_bandwidth='scott')
    assert mutual_corr >= 0
    assert mutual_corr > mutual_uncorr
