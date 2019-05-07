'''Computation of mutual information'''

import numpy as np

from scipy.special import gamma,psi
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors
from scipy.stats.kde import gaussian_kde

# __all__=['entropy', 'mutual_information', 'entropy_gaussian']
#
# EPS = np.finfo(float).eps


def _nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions

    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def _entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(det(C)))


def _entropy(X, k=1):
    ''' Returns the entropy of the X.

    Parameters
    ===========

    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed

    k : int, optional
        number of nearest neighbors for density estimation

    Notes
    ======

    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = _nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.

    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions

    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation

    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([_entropy(X, k=k) for X in variables])
            - _entropy(all_vars, k=k))


def mutual_kde(x, y, type_bandwidth='scott'):
    """Mutual information estimated as cumulative sum  of ratio
     P(x)P(y)/P(x,y)
     The probability density functions we estimate with kernel-dencity
     estimator (KDE) using Gaussian kernels.

    :param x: numpy.array, shape (n_samples)
    :param y: numpy.array, shape (n_samples)
    :param type_bandwidth: str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', '2scott', '05scott'
    :return: float
        Mutual information, a non-negative value

    Notes:
    Bandwidth make influence on the KDE estimation. We use Scott's rule,
    'scott', that is default paraeter in 'gaussian_kde'
    """


    xmin = x.min() - 0.1 * (x.max() - x.min())
    xmax = x.max() + 0.1 * (x.max() - x.min())
    ymin = y.min() - 0.1 * (y.max() - y.min())
    ymax = y.max() + 0.1 * (y.max() - y.min())

    Xm, Ym = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    XYm = np.c_[Xm.ravel(), Ym.ravel()]

    xy = np.c_[x, y]
    bandwidth = 'scott'
    kde_xy = gaussian_kde(xy.T, bw_method=bandwidth)
    kde_x = gaussian_kde(x.ravel(), bw_method=bandwidth)
    kde_y = gaussian_kde(y.ravel(), bw_method=bandwidth)

    if type_bandwidth == '2scott':
        kde_xy.set_bandwidth(bw_method=bandwidth)
        kde_xy.set_bandwidth(bw_method=kde_xy.factor * 2.0)

        kde_x.set_bandwidth(bw_method=bandwidth)
        kde_x.set_bandwidth(bw_method=kde_x.factor * 2.0)

        kde_y.set_bandwidth(bw_method=bandwidth)
        kde_y.set_bandwidth(bw_method=kde_y.factor * 2.0)

    if type_bandwidth == '05scott':
        kde_xy.set_bandwidth(bw_method=bandwidth)
        kde_xy.set_bandwidth(bw_method=kde_xy.factor * 0.5)

        kde_x.set_bandwidth(bw_method=bandwidth)
        kde_x.set_bandwidth(bw_method=kde_x.factor * 0.5)

        kde_y.set_bandwidth(bw_method=bandwidth)
        kde_y.set_bandwidth(bw_method=kde_y.factor * 0.5)

    # Mutual information
    kde_xy_values = kde_xy(XYm.T)
    mutual_information = np.sum(kde_xy_values *
                                (np.log((kde_xy_values /
                                         (kde_x(Xm[:, 0])[:, np.newaxis] *
                                          kde_y(Ym[0])).ravel())
                                        ) + 1e-4)
                                )
    return mutual_information

