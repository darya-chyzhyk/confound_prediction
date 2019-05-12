'''Deconfounding
1. Confound izolation cross validation
2. Confound regress-out
    a. jointly
    b. out-of-sample
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression

from scipy import linalg

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



def confound_isolating_cv(X, y, ids_sampled):
    """
    Function that create the test and training sets, masking the samples with
    indexes obtained from the Confound Isolation sampling

    :param X: array-like, shape (n_samples, n_features)
    :param y: array-like, shape (n_samples)
    :param ids_sampled: list
    :return: list of arrays,
        train and test of X, y and ids
    """

    x_test = []
    x_train = []
    y_test = []
    y_train = []
    ids_test = []
    ids_train = []

    ids = list(range(0, y.shape[0]))

    for index_list in ids_sampled:
        mask = np.isin(ids, index_list)
        x_test.append(X[mask])
        x_train.append(X[~mask])
        y_test.append(y[mask])
        y_train.append(y[~mask])
        ids_test.append(index_list)
        ids_train.append(np.array(ids)[~mask])
    return x_test, x_train, y_test, y_train, ids_test, ids_train





def deconfound_jointly(signals, confounds):
    """
    Adapted code from the Nilern code

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

    # Remove confounds
    # TODO create _ensure_float function
    # confounds = _ensure_float(confounds)
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

    return signals


def confound_regressout(X, y, z, ids_sampled, type_deconfound):
    # Create test and train sets
    x_test = []
    x_train = []
    y_test = []
    y_train = []
    ids_test = []
    ids_train = []


    ids = list(range(0, y.shape[0]))

    for index_list in ids_sampled:
        mask = np.isin(ids, index_list)
        # test

        y_test.append(y[mask])
        ids_test.append(index_list)
        #z_conf_test.append(z_conf[mask])
        # train

        y_train.append(y[~mask])
        ids_train.append(np.array(ids)[~mask])
        #z_conf_train.append(z_conf[~mask])


        if type_deconfound is 'out_of_sample':

            # train test


            deconfounder = DeConfounder()
            deconfounder.fit(X[~mask], z[~mask])
            x_test.append(deconfounder.transform(X[mask], z[mask]))
            x_train.append(X[~mask])



        elif (type_deconfound is 'jointly') or (type_deconfound is
                                                       'False'):
            x_test.append([mask])
            x_train.append(X[~mask])




    name_base = ('Simulation_exp1_' + type_sampling
                 + '_regressout_' + str(do_conf_regressout)
                 + '_permutations_' + str(n_permutations)
                 + '_seeds_' + str(n_seeds))

    if (n_permutations is None) or (n_permutations == 0):
        name_csv_prediction = (name_base + '.csv')
        # prediction y from X, z
        results = prediction_uni_out_given_datasplit(
            x_train, x_test, y_train, y_test, ids_train, ids_test,
            regression_list, results_path, dataset_name, predict_name,
            atlas, con_measure, sampling_name=type_sampling,
            confounds_name=confounds_name,
            do_conf_regressout=do_conf_regressout, n_jobs=n_jobs,
            to_csv=True,
            name_csv_prediction=name_csv_prediction)

    else:
        results = prediction_with_permutation(
            x_train, x_test, y_train, y_test, ids_train, ids_test,
            regression_list, results_path, dataset_name,
            predict_name, atlas, con_measure,
            sampling_name=type_sampling, confounds_name=confounds_name,
            do_conf_regressout=do_conf_regressout, n_jobs=n_jobs,
            to_csv=True, n_permutations=n_permutations, name_base=name_base)

    # save sampled info
    save_timeseries_to_pkl(sampled_set, results_path,
                           name_file=name_base,
                           suffix='_sampled',
                           extension=None)
