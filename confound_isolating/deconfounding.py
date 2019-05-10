'''Deconfounding
1. Confound izolation cross validation
2. Confound regress-out
    a. jointly
    b. separatly
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression

class DeConfounder(BaseEstimator, TransformerMixin):
    """ A transformer removing the effect of y on X.
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



def confound_isolating_cv(X, y, ids_sampled, n_permutations=None):

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

        x_test.append(X[mask])
        x_train.append(X[~mask])




    # name_base = ('Simulation_exp1_' + type_sampling
    #              + '_regressout_' + str(do_conf_regressout)
    #              + '_permutations_' + str(n_permutations)
    #              + '_seeds_' + str(n_seeds))

    if (n_permutations is None) or (n_permutations == 0):
        # name_csv_prediction = (name_base + '.csv')
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
elif predict_name == 'y_from_z':
    #######################################################################
    # Prediction from confounds
    name_base = ('Simulation_exp1_' + type_sampling
                 + '_regressout_' + str(do_conf_regressout)
                 + '_permutations_' + str(n_permutations)
                 + '_seeds_' + str(n_seeds))

    name_csv_prediction = (name_base + '.csv')

    # Cross validation
    n_iter = 10
    cv = ShuffleSplit(n_splits=n_iter, test_size=0.2, random_state=0)
    # Predict
    if (n_permutations == 0) or (n_permutations == None):

        prediction_uni_out(y.reshape(-1,1), z_conf, ids, cv,
                           regression_list,
                           results_path,
                           dataset_name, predict_name, 'atlasNone',
                           'cmNone',
                           groups=None, n_jobs=n_jobs, to_csv=True,
                           name_csv_prediction=name_csv_prediction)
    else:

        cv_split = cv.split(y.reshape(-1,1), z_conf)
        n_folds = cv.get_n_splits(y.reshape(-1,1), z_conf)  # number of folds in cv

        x_train = []
        x_test = []
        y_train = []
        y_test = []
        ids_train = []
        ids_test = []
        ids = np.array(ids)
        for train_index, test_index in cv_split:
            x_train.append(y[train_index])
            x_test.append(y[test_index])
            y_train.append(z_conf[train_index])
            y_test.append(z_conf[test_index])
            ids_train.append(ids[train_index])
            ids_test.append(ids[test_index])

        # if type(ids) is not np.ndarray:
        #     print(type(ids))
        #     ids = np.array(ids)
        #     print(type(ids))
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        results = prediction_with_permutation(
            x_train,
            x_test, y_train,
            y_test,
            ids_train, ids_test,
            regression_list, results_path, dataset_name,
            predict_name, atlas, con_measure,
            sampling_name=None, confounds_name=confounds_name,
            do_conf_regressout=do_conf_regressout, n_jobs=n_jobs,
            to_csv=True, n_permutations=n_permutations, name_base=name_base)



def confound_regressout(y, type_deconfound, ids_sampled):
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

        if do_conf_regressout is 'separately':
            x_test.append(clean(X_brain[mask], standardize=False, detrend=False,
                           confounds=z_conf[mask], low_pass=None,
                           high_pass=None))

            x_train.append(clean(X_brain[~mask], standardize=False,
                                 detrend=False,
                            confounds=z_conf[~mask], low_pass=None,
                            high_pass=None))
        elif do_conf_regressout is 'model_confounds':

            # train test


            deconfounder = DeConfounder()
            deconfounder.fit(X_brain[~mask], z_conf[~mask])
            x_test.append(deconfounder.transform(X_brain[mask], z_conf[mask]))
            x_train.append(X_brain[~mask])

            if deconfound_y is True:
                y = clean(y, standardize=False, detrend=False,
                          confounds=z_conf, low_pass=None,
                          high_pass=None)

        elif (do_conf_regressout is 'jointly') or (do_conf_regressout is
                                                       'False'):
            x_test.append(X_brain[mask])
            x_train.append(X_brain[~mask])




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
