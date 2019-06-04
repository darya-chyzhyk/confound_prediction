"""
Sampling methods

"""

import numpy as np
from scipy.stats.kde import gaussian_kde

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets.base import Bunch

from confound_prediction.mutual_information import mutual_kde
from confound_prediction.utils import _ensure_int_positive

# TODO
# sampling_with_kde
# split to
# 1) confound isolation
# 2) randome


def confound_isolating_index_2remove(y, z, n_remove=None, prng=None):
    """
    The goal is to find a test set with independence between y and z

    :param y: numpy.array, shape (n_samples), target
    :param z: numpy.array, shape (n_samples), confound
    :param n_remove: int,
        number to be removed in each iteration, the default is 4
    :param prng: np.random.RandomState, default is None
        control the pseudo random number generator
    :return: numpy.array, shape (m_samples),
        index to be removed, m < n
    """

    n_remove = _ensure_int_positive(n_remove, default=4)

    y_train, y_test, z_train, z_test, index_train, index_test = \
        train_test_split(y, z, np.arange(y.shape[0]), test_size=0.25,
                         random_state=42)

    yz_train = np.array((y_train, z_train))
    yz_test = np.array((y_test, z_test))

    # Scaling for kde
    bandwidth = 'scott'
    scaler = preprocessing.StandardScaler()
    scaler.fit(yz_train.T)

    train_scaled = scaler.transform(yz_train.T).T
    test_scaled = scaler.transform(yz_test.T).T
    kde_yz = gaussian_kde(train_scaled, bw_method=bandwidth)

    # Bandwidth: train and test on kde_yz, and use it for kde_y and kde_z
    bandwidth_xy = kde_yz.factor
    kde_y = gaussian_kde(train_scaled[0], bw_method=bandwidth_xy)
    kde_z = gaussian_kde(train_scaled[1], bw_method=bandwidth_xy)

    # Ratio, to find max
    ratio_dens = (kde_yz(test_scaled)) / (kde_y(test_scaled[0]) * kde_z(
        test_scaled[1]))  # + 1e-4

    # Subjects to remove
    index_sort = np.argsort(ratio_dens)
    ratio_sort = ratio_dens[np.argsort(ratio_dens)]
    empirical_cdf = (np.cumsum(ratio_sort)) ** 5

    # TODO add parameters of number of discarded subjects, at the moment is
    #  a constant = 4

    if prng is None:
        random_quantiles = np.random.random(size=n_remove) * empirical_cdf.max()
    else:
        random_quantiles = prng.rand(n_remove) * empirical_cdf.max()
    idx_to_reject = np.searchsorted(empirical_cdf, random_quantiles,
                                    side='left')
    # Index from test subset to be removed
    index_to_remove = index_test[index_sort[idx_to_reject]]

    # TODO remove repetitions in the index_to_remove, but not sure its
    #  possible or important

    return index_to_remove


def random_index_2remove(y, z, n_remove=None):
    """
    Function to select 4 random indexes to remove
    :param y: numpy.array, shape (n_samples), target
    :param z: numpy.array, shape (n_samples), confound
    :param n_remove: int,
        number to be removed in each iteration, the default is 4
    :return: numpy.array, shape (m_samples),
        index to be removed, m < n
    """
    n_remove = _ensure_int_positive(n_remove, default=4)

    y_train, y_test, z_train, z_test, index_train, index_test = \
        train_test_split(y, z, np.arange(y.shape[0]), test_size=0.25,
                         random_state=42)

    if index_test.shape[0] >= n_remove:
        index_to_remove = np.random.choice(index_test, n_remove, replace=False)
    else:
        index_to_remove = np.array([])

    # TODO make index_to_remove integer
    # index_to_remove = np.random.randint(index_test, 4, replace=False)
    # TODO for the output keep just index_to_remove

    # TODO make number of removing samples (4) as parameter?

    return index_to_remove


def confound_isolating_sampling(y, z, random_seed=None, min_sample_size=None,
                                n_remove=None):
    """
    Sampling method based on the 'Confound isolating cross-validation'
    technique.
    # TODO Reference to the paper

    :param y: numpy.array, shape (n_samples), target
    :param z: numpy.array, shape (n_samples), confound
    :param random_seed: int
        Random seed used to initialize the pseudo-random number generator.
        Can be any integer between 0 and 2**32 - 1 inclusive. Defaul is None
    :param min_sample_size: int
        Minimum sample size (in samples) to be reached, default is 10% of the
        data
    :param n_remove: int,
        number of the samples to be removed on each iteration of sampling,
        default is 4
    :return:
        sampled_index,
        mutual_information
        correlation
    """

    sampled_index = list(range(0, y.shape[0]))
    mutual_information = []
    correlation = []
    index_to_remove = []

    n_remove = _ensure_int_positive(n_remove, default=4)

    min_sample_size = _ensure_int_positive(min_sample_size, default=10)
    min_size = np.int(y.shape[0] * min_sample_size / 100)

    while y.shape[0] > min_size:

        # remove subject from the previous iteration
        y = np.delete(y, index_to_remove, axis=0)
        z = np.delete(z, index_to_remove, axis=0)
        sampled_index = np.delete(sampled_index, index_to_remove, axis=0)

        # control the pseudo random number generator
        if random_seed is None:
            prng = None
        else:
            prng = np.random.RandomState(seed=random_seed)

        # return indexes
        index_to_remove = confound_isolating_index_2remove(y, z,
                                                           n_remove=n_remove,
                                                           prng=prng)

        # The case when target and confound are equal
        if np.all(y==z) == True:
            mutual_information.append('NaN')
        else:
            mutual_information.append(mutual_kde(y.astype(float),
                                                 z.astype(float)))
        correlation.append(np.corrcoef(y.astype(float), z.astype(float))[0, 1])

    # sampled_set = {'sampled_index': array_data[:, 2],
    #                'mutual_information': mi_list,
    #                'correlation': corr_list}
    # sampled_index = array_data[:, 2]
    # return Bunch(**sampled_set)
    return sampled_index, mutual_information, correlation


def random_sampling(y, z, min_sample_size=None, n_remove=None):
    """
    :param y: numpy.array, shape (n_samples), target
    :param z: numpy.array, shape (n_samples), confound
    :param n_remove: int,
        number of the samples to be removee on each itteration of sampling,
        default is 4
    :param min_sample_size: float
        Minimum sample size to be reached, default is 10% of the data
    :return:
    """

    sampled_index = list(range(0, y.shape[0]))
    mutual_information = []
    correlation = []
    index_to_remove = []
    no_index = 0 # rin case of the size of test set inside of the
    # 'random_index_2remove' is smaller then 'min_sample_size', no_index
    # becomes 1 and stop the sampling

    n_remove = _ensure_int_positive(n_remove, default=4)

    min_sample_size = _ensure_int_positive(min_sample_size, default=10)
    min_size = np.int(y.shape[0] * min_sample_size / 100)

    while (y.shape[0] > min_size) and (no_index == 0):

        # remove subject from the previous iteration
        y = np.delete(y, index_to_remove, axis=0)
        z = np.delete(z, index_to_remove, axis=0)
        sampled_index = np.delete(sampled_index, index_to_remove, axis=0)

        # return indexes
        index_to_remove = random_index_2remove(y, z, n_remove=n_remove)
        if index_to_remove.shape[0] == 0:
            no_index = 1

        # The case when target and confound are equal
        if np.all(y == z) == True:
            mutual_information.append('NaN')
        else:
            mutual_information.append(mutual_kde(y.astype(float),
                                                 z.astype(float)))

        correlation.append(np.corrcoef(y.astype(float), z.astype(float))[0, 1])

        # sampled_set = {'sampled_index': array_data[:, 2],
        #                'mutual_information': mi_list,
        #                'correlation': corr_list}
        # sampled_index = array_data[:, 2]

    #return Bunch(**sampled_set)
    return sampled_index, mutual_information, correlation



############################################################################
# Delete

def sampling_with_kde(x, y, type_sampling, prng=None):
    # TODO split into 2 functions
    # confound_isolating_sampling
    '''

    :param x: numpy.array
    :param y: numpy.array
    :param type_sampling: 'sampling_cumsum' - with kde,
    :param prng:
    :return:
    '''
    bandwidth = 'scott'
    x_train, x_test, y_train, y_test, index_train, index_test = \
        train_test_split(x, y, np.arange(x.shape[0]), test_size=0.25,
                         random_state=42)

    xy_train = np.array((x_train, y_train))
    xy_test = np.array((x_test, y_test))

    # Options for sampling

    if type_sampling == 'sampling_cumsum':
        # scaling for kde
        scaler = preprocessing.StandardScaler()
        scaler.fit(xy_train.T)
        scale_method = scaler
        train_scaled = scaler.transform(xy_train.T).T
        test_scaled = scaler.transform(xy_test.T).T
        kde_xy = gaussian_kde(train_scaled, bw_method=bandwidth)

        # bandwidth: train and test on kde_xy, and use it for kde_x and kde_y
        bandwidth_xy = kde_xy.factor
        kde_x = gaussian_kde(train_scaled[0], bw_method=bandwidth_xy)
        kde_y = gaussian_kde(train_scaled[1], bw_method=bandwidth_xy)

        # Ratio, to find max
        ratio_dens = (kde_xy(test_scaled)) / (kde_x(test_scaled[0]) * kde_y(
            test_scaled[1]))  # + 1e-4


    elif type_sampling == 'random_4':
        ratio_dens, kde_x, kde_y, kde_xy, scale_method = [0,0,0,0,0]

    # subjects to remove
    if type_sampling == 'random_4':
        index_test_remove = np.random.choice(index_test, 4, replace=False)
    else:
        # index_sort = np.argsort(-ratio_dens)
        # ratio_sort = ratio_dens[np.argsort(-ratio_dens)]

        index_sort = np.argsort(ratio_dens)
        ratio_sort = ratio_dens[np.argsort(ratio_dens)]

        empirical_cdf = (np.cumsum(ratio_sort)) ** 5
        if prng is None:
            random_quantiles = np.random.random(size=4) * empirical_cdf.max()
        else:
            random_quantiles = prng.rand(4) * empirical_cdf.max()
        idx_to_reject = np.searchsorted(empirical_cdf, random_quantiles,
                                        side='left')
        ratio_remove = ratio_dens[index_sort[idx_to_reject]]
        # index from test subset to remove
        index_test_remove = index_test[index_sort[idx_to_reject]]
    #print('Index in x_test to remove: ', index_test_remove)
    #mask[index_test_remove] = 0
    return ratio_dens, index_test_remove, kde_x, kde_y, kde_xy, scale_method



# TODO move to example


def confound_izolating_sampling_iterations(y, z, n_seed=0,
             min_size=None, save_mi_iter=False, type_bandwidth='scott'):

    # The same function as 'confound_izolating_sampling', but saving etterations

    '''
    Sampling
    :param y:
    :param z:
    :param ids:
    :param type_sampling:
    :param n_seeds:
    :param min_size: float
        size of sapled test, default is 10% of the data
    :return:
    '''
    ids = list(range(0, y.shape[0]))

    mi_list = []
    corr_list = []
    mi_iter = []
    corr_iter = []
    seed_iter = []
    n_subjects_list = []
    y_sampling_list = []
    z_sampling_list = []

    n_iter = 0
    index_to_remove = []
    array_data = np.c_[y, z, ids]
    if min_size is None:
        min_size = np.int(y.shape[0] / 10)
    else:
        min_size = np.int(y.shape[0] * min_size / 100)

    while array_data.shape[0] > min_size:

        n_iter = n_iter + 1
        # remove subject from the previous iteration
        array_data = np.delete(array_data, index_to_remove, axis=0)
        y_sampling = array_data[:, 0]
        z_sampling = array_data[:, 1]

        # control the pseudo random number generator
        prng = np.random.RandomState(seed=n_seed)

        # return indexes
        index_to_remove = confound_isolating_index_2remove(y, z, prng=None)
        # ratio_dens, index_to_remove, kde_y, kde_z, kde_yz, scale_method = \
        #     sampling_with_kde(y_sampling, z_sampling, type_sampling, prng)

        if save_mi_iter is True:

            # save sampled info
            mi_iter.append(mutual_kde(y_sampling.astype(float),
                                      z_sampling.astype(float),
                                      type_bandwidth=type_bandwidth))
            corr_iter.append(np.corrcoef(y_sampling.astype(float),
                                         z_sampling.astype(float))[0, 1])
            seed_iter.append(n_seed)
            n_subjects_list.append(array_data.shape[0])
            y_sampling_list.append(y_sampling)
            z_sampling_list.append(z_sampling)

    # Mutual information
    # MI iterration
    if save_mi_iter is True:

        sampled_iter = {'n_seeds': seed_iter,
                        'n_subjects': n_subjects_list,
                        'mutual': mi_iter,
                        'correlation': corr_iter,
                        'Sampling': len(seed_iter) * [type_sampling],
                        'y_sampling_list': y_sampling_list,
                        'z_sampling_list': z_sampling_list}
    else:
        sampled_iter = {}


    if np.all(y_sampling==z_sampling) == True:
        mi_list.append('NaN')
    else:
        mi_list.append(mutual_kde(y_sampling.astype(float),
                                  z_sampling.astype(float),
                                  type_bandwidth=type_bandwidth))
    corr_list.append(np.corrcoef(y_sampling.astype(float),
                                 z_sampling.astype(float))[0, 1])


    sampled_set = {'n_seeds': n_seed,
                   'ids_sampled': array_data[:, 2],
                   'mutual': mi_list,
                   'correlation': corr_list}


    return Bunch(**sampled_set), Bunch(**sampled_iter)
