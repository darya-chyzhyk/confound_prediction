'''Deconfounding
1. Confound isolation cross validation
2. Confound regress-out
    a. jointly
    b. out-of-sample
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression

from scipy import linalg

from confound_prediction.sampling import (random_sampling,
                                          confound_isolating_sampling)

class DeConfounder(BaseEstimator, TransformerMixin):
    """ A transformer removing the effect of y on X using
    sklearn.linear_model.LinearRegression.
    """

    def __init__(self, confound_model=LinearRegression()):
        self.confound_model = confound_model

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        confound_model = clone(self.confound_model)
        confound_model.fit(y, X)
        self.confound_model_ = confound_model

        return self

    def transform(self, X, y):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        X_confounds = self.confound_model_.predict(y)
        return X - X_confounds


def confound_isolating_cv(X, y, z, random_seed=None, min_sample_size=None,
                          cv_folds=10, n_remove=None):
    """
    Function that create the test and training sets, masking the samples with
    indexes obtained from the Confound Isolation sampling

    :param X: array-like, shape (n_samples, n_features)
    :param y: array-like, shape (n_samples), target
    :param z: numpy.array, shape (n_samples), confound: list
    :param random_seed: int
        Random seed used to initialize the pseudo-random number generator.
        Can be any integer between 0 and 2**32 - 1 inclusive. Default is None
    :param min_sample_size: int
        Minimum sample size (in samples) to be reached, default is 10% of the
        data
    :param cv_folds: int
        number of folders to mimic the cross validation, default is 10.
    :param n_remove: int,
        number of the samples to be removee on each itteration of sampling,
        default is 4

    :return: list of arrays,
        train and test of X, y and sampled indexes
    """
    x_test = []
    x_train = []
    y_test = []
    y_train = []
    ids_test = []
    ids_train = []
    ids_sampled = []

    # Sampling
    for cv_fold in range(cv_folds):
        ids_sampled_fold, _, _ = \
            confound_isolating_sampling(y, z, random_seed=random_seed,
                                        min_sample_size=min_sample_size,
                                        n_remove = n_remove)
        ids_sampled.append(ids_sampled_fold)

    for index_list in ids_sampled:
        ids = list(range(0, y.shape[0]))
        mask = np.isin(ids, index_list)
        x_test.append(X[mask])
        x_train.append(X[~mask])
        y_test.append(y[mask])
        y_train.append(y[~mask])
        ids_test.append(index_list)
        ids_train.append(np.array(ids)[~mask])
    return x_test, x_train, y_test, y_train, ids_test, ids_train


def deconfound_model_jointly(signals, confounds):
    """
    Adapted code from the Nilern.signal.clean code for deconfounding jointly

    :param signals: numpy.ndarray
        Timeseries. Must have shape (instant number, features number).
    :param confounds:numpy.ndarray or list of Confounds timeseries.
        Shape must be (instant number, confound number), or just
        (instant number,) The number of time instants in signals and confounds
        must be identical (i.e. signals.shape[0] == confounds.shape[0]).
        If a list is provided, all confounds are removed from the input signal,
        as if all were in the same array.
    :return: numpy.ndarray
        Input signals, deconfounded. Same shape as signals.
    """

    # TODO create _ensure_float function
    # confounds = _ensure_float(confounds)

    # Remove confounds
    if not isinstance(confounds, (list, tuple)):
        confounds = (confounds,)

    all_confounds = []
    for confound in confounds:
        if isinstance(confound, np.ndarray):
            if confound.ndim == 1:
                confound = np.atleast_2d(confound).T
            elif confound.ndim != 2:
                raise ValueError("confound array has an incorrect number "
                                 "of dimensions: %d" % confound.ndim)
            if confound.shape[0] != signals.shape[0]:
                raise ValueError("Confound signal has an incorrect length")
        else:
            raise TypeError("confound has an unhandled type: %s"
                            % confound.__class__)
        all_confounds.append(confound)

    # Restrict the signal to the orthogonal of the confounds
    confounds = np.hstack(all_confounds)
    del all_confounds

    # Improve numerical stability by controlling the range of
    # confounds. We don't rely on _standardize as it removes any
    # constant contribution to confounds.
    confound_max = np.max(np.abs(confounds), axis=0)
    confound_max[confound_max == 0] = 1
    confounds /= confound_max

    # Pivoting in qr decomposition was added in scipy 0.10
    Q, R, _ = linalg.qr(confounds, mode='economic', pivoting=True)
    Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100.]
    signals -= Q.dot(Q.T).dot(signals)
    signals_deconfounded = np.copy(signals)
    return signals_deconfounded


def confound_regressout(X, y, z, type_deconfound='out_of_sample',
                        min_sample_size=None, cv_folds=10, n_remove=None):
    """

    :param X: array-like, shape (n_samples, n_features)
    :param y: array-like, shape (n_samples), target
    :param z: numpy.array, shape (n_samples), confound: list
    :param type_deconfound: str,
        # TODO correct the possible options
        The possible options are 'jointly', 'out_of_sample' and
        'False'. The default is 'out_of_sample'
    :param min_sample_size: float
        Minimum sample size to be reached, default is 10% of the data
    :param n_remove: int,
        number of the samples to be removeed on each iteration of sampling,
        default is 4
    :param cv_folds: int
        number of folders for k-cross validation
    :return: list of numpy.ndarray
        Deconfounded and split 'X' and 'y' to the test and train data with
        the indexes test.
    """
    # TODO decide the name of of options for 'type_deconfound'
    # Jointly
    # Out - of - sample
    # Create test and train sets
    x_test = []
    x_train = []
    y_test = []
    y_train = []
    ids_test = []
    ids_train = []
    ids_sampled = []

    # Pre-confounding
    if type_deconfound == 'jointly':
        X = deconfound_model_jointly(X, z)

    # Sampling
    for cv_fold in range(cv_folds):
        ids_sampled_fold, _, _ = random_sampling(y, z,
                                                 min_sample_size=min_sample_size,
                                                 n_remove=n_remove)
        ids_sampled.append(ids_sampled_fold)

    for index_list in ids_sampled:

        ids = list(range(0, y.shape[0]))
        mask = np.isin(ids, index_list)

        # Creating test and train
        y_test.append(y[mask])
        y_train.append(y[~mask])
        ids_test.append(np.array(ids)[mask])
        ids_train.append(np.array(ids)[~mask])

        # Deconfound
        if type_deconfound is 'out_of_sample':

            deconfounder = DeConfounder()
            deconfounder.fit(X[~mask], z[~mask])
            x_test.append(deconfounder.transform(X[mask], z[mask]))
            x_train.append(X[~mask])

        elif (type_deconfound is 'jointly') or (type_deconfound is 'False'):
            x_test.append(X[mask])
            x_train.append(X[~mask])
    return x_test, x_train, y_test, y_train, ids_test, ids_train

