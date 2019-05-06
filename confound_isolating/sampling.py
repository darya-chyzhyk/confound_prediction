"""
Sampling

"""

import numpy as np
from scipy.stats.kde import gaussian_kde

def mutual_kde(x, y, type_bandwidth='scott'):
    '''

    :param x:
    :param y:
    :param type_bandwidth:
    :return:
    '''
    bandwidth = 'scott'
    xmin = x.min() - 0.1 * (x.max() - x.min())
    xmax = x.max() + 0.1 * (x.max() - x.min())
    ymin = y.min() - 0.1 * (y.max() - y.min())
    ymax = y.max() + 0.1 * (y.max() - y.min())

    Xm, Ym = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    XYm = np.c_[Xm.ravel(), Ym.ravel()]

    xy = np.c_[x, y]

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
