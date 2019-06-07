import os
import pandas as pd
from collections import OrderedDict
import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             explained_variance_score, r2_score)
from sklearn.multioutput import MultiOutputRegressor

from sklearn.utils import shuffle


from joblib import Parallel, delayed



def prediction_uni_out_given_datasplit(x_train_list, x_test_list, y_train_list,
                                       y_test_list, ids_train_list,
                                       ids_test_list, regr_uni_list,
                                       results_path, dataset_name,
                                       predict_name, atlas,
                                       connectivity_measure,
                                       sampling_name=None, confounds_name=None,
                                       do_conf_regressout='True',
                                       n_jobs=10, to_csv=True,
                                       name_csv_prediction=False):

    """ Uni-output prediction with given data split to train and test

    Parameters
    ----------
    x: array,
        Training vector, array-like, shape (n_samples, n_features)
    y:  array,
        Target vector, array-like, shape (n_samples)
    test_index: list
         list of index for test, e.i. from sampling function
    regr_uni_list: list, str
        list of uni output regressors
    results_path: str,
        path to the result folder
    dataset_name: str
        name of the dataset, i.e. 'HCP'.
    predict_name: str,
        name of the scores to predict, i.e. 'Age'.
    atlas: str
        name of the atlas
    connectivity_measure: str
        name of the connectivity measure, i.e. 'tangent'
    groups: array,
        default is None,
    n_jobs: int,
        number of jobs, default is 10,
    to_csv: boolean,
        True if save to csv, defaut is False
    name_csv_prediction: str,
        name of the file to save

    """

    name_regr_list = []
    mse_list = []
    mae_list = []
    evs_list = []
    r2s_list = []


    mse_std_list = []
    mae_std_list = []
    evs_std_list = []
    r2s_std_list = []

    y_test_array = []
    y_predict_array = []

    # testing on the same TRAIN set
    mse_list_train = []
    mae_list_train = []
    evs_list_train = []
    r2s_list_train = []

    mse_std_list_train = []
    mae_std_list_train = []
    evs_std_list_train = []
    r2s_std_list_train = []

    # save fold umber

    for regression_model in regr_uni_list:
        name_regr = str(regression_model)[0:str(regression_model).find('(')]
        if name_regr == 'SVR':
            name_regr = name_regr + '_' + regression_model.kernel

        name_regr_list.append(name_regr)
        subj_number_unique = x_train_list[0].shape[0] + x_test_list[0].shape[0]
        print(name_regr)
        n_folds = len(x_train_list)

        list_n_train_test = []
        for x_train, x_test, y_train, y_test, ids_train, ids_test, n \
                in zip(x_train_list, x_test_list, y_train_list, y_test_list,
                       ids_train_list, ids_test_list,
                       np.array(range(1, n_folds + 1))):

            # in case x_train is a list
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            # in case of prediction from confound, the size of confounds is (n,)
            if len(x_train.shape) == 1:
                x_train = x_train.reshape(-1, 1)
            if len(x_test.shape) == 1:
                x_test = x_test.reshape(-1, 1)

            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)





            list_n_train_test.append([x_train, x_test, y_train, y_test,
                                      ids_train, ids_test, n])

        # test with and without parallel
        if (n_jobs == 0) or (n_jobs is None):
            fit_result = []


            for x_train, x_test, y_train, y_test, ids_train, ids_test, n \
                    in list_n_train_test:
                #print(n)

                fit_result.append(model_fit_datasplit(x_train, x_test, y_train,
                                                y_test,
                ids_train, ids_test, regression_model, n))
        else:
            fit_result = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(model_fit_datasplit)(x_train, x_test, y_train, y_test,
                                             ids_train, ids_test,
                                             regression_model, n)
                for x_train, x_test, y_train, y_test, ids_train, ids_test,
                    n in list_n_train_test)

        y_test_array = []
        y_predict_array = []
        mse_list_cv = []
        mae_list_cv = []
        evs_list_cv = []
        r2s_list_cv = []
        ids_train = []
        ids_test = []
        mse_list_cv_train = []
        mae_list_cv_train = []
        evs_list_cv_train = []
        r2s_list_cv_train = []
        n_fold = []
        regression_model_fit = []

        for dic in fit_result:
            y_test_array.append(dic['y_test'])
            y_predict_array.append(dic['regr_predict'])

            mse_list_cv.append(dic['mse_test'])
            mae_list_cv.append(dic['mae_test'])
            evs_list_cv.append(dic['evs_test'])
            r2s_list_cv.append(dic['r2s_test'])

            ids_train.append(dic['ids_train'])
            ids_test.append(dic['ids_test'])

            mse_list_cv_train.append(dic['mse_train'])
            mae_list_cv_train.append(dic['mae_train'])
            evs_list_cv_train.append(dic['evs_train'])
            r2s_list_cv_train.append(dic['r2s_train'])

            n_fold.append(dic['cvfold_number'])
            regression_model_fit.append(dic['regression_model'])

        # cv folds results
        datasetname_list_cv = [dataset_name] * len(mse_list_cv)
        predict_list_cv = [predict_name] * len(mse_list_cv)
        atlas_list_cv = [atlas] * len(mse_list_cv)
        connectivity_list_cv = [connectivity_measure] * len(mse_list_cv)
        name_regr_list_cv = [name_regr] * len(mse_list_cv)
        sampling_name_list_cv = [sampling_name] * len(mse_list_cv)
        confounds_name_list_cv = [confounds_name] * len(mse_list_cv)
        do_conf_regressout_list_cv = [do_conf_regressout] * len(mse_list_cv)

        subj_number_cv = [subj_number_unique] * len(mse_list_cv)
        scan_number_cv = [subj_number_unique] * len(mse_list_cv)

        dict_prediction_uni_cvfold = {'Dataset': datasetname_list_cv,
                                      'Predict': predict_list_cv,
                                      'Atlas': atlas_list_cv,
                                      'Connectivity': connectivity_list_cv,
                                      'Model': name_regr_list_cv,
                                      'Sampling': sampling_name_list_cv,
                                      'Confounds': confounds_name_list_cv,
                                      'Regressout': do_conf_regressout_list_cv,
                                      'Subj_number': subj_number_cv,
                                      'Scan_number': scan_number_cv,
                                      'CvFold_number': n_fold,
                                      'MSE_cv_test': mse_list_cv,
                                      'MAE_cv_test': mae_list_cv,
                                      'EVS_cv_test': evs_list_cv,
                                      'R2S_cv_test': r2s_list_cv,
                                      'MSE_cv_train': mse_list_cv_train,
                                      'MAE_cv_train': mae_list_cv_train,
                                      'EVS_cv_train': evs_list_cv_train,
                                      'R2S_cv_train': r2s_list_cv_train}

        mse_list.append(np.mean(mse_list_cv))
        mae_list.append(np.mean(mae_list_cv))
        evs_list.append(np.mean(evs_list_cv))
        r2s_list.append(np.mean(r2s_list_cv))

        mse_std_list.append(np.std(mse_list_cv))
        mae_std_list.append(np.std(mae_list_cv))
        evs_std_list.append(np.std(evs_list_cv))
        r2s_std_list.append(np.std(r2s_list_cv))

        # TRAIN set
        mse_list_train.append(np.mean(mse_list_cv_train))
        mae_list_train.append(np.mean(mae_list_cv_train))
        evs_list_train.append(np.mean(evs_list_cv_train))
        r2s_list_train.append(np.mean(r2s_list_cv_train))

        mse_std_list_train.append(np.std(mse_list_cv_train))
        mae_std_list_train.append(np.std(mae_list_cv_train))
        evs_std_list_train.append(np.std(evs_list_cv_train))
        r2s_std_list_train.append(np.std(r2s_list_cv_train))

        datasetname_list_y = [dataset_name] * len(y_test_array)
        predict_list_y = [predict_name] * len(y_test_array)
        atlas_list_y = [atlas] * len(y_test_array)
        connectivity_list_y = [connectivity_measure] * len(y_test_array)
        name_regr_list_y = [name_regr] * len(y_test_array)

        sampling_namer_list_y = [sampling_name] * len(y_test_array)
        confounds_name_list_y = [confounds_name] * len(y_test_array)
        do_conf_regressout_list_y = [do_conf_regressout] * len(y_test_array)

        subj_number_y = [subj_number_unique] * len(y_test_array)
        scan_number_y = [subj_number_unique] * len(y_test_array)
        # scan_number_y = [x.shape[0]] * len(y_test_array)

        df_y_test_y_prediction = pd.DataFrame(
        OrderedDict((('Dataset', datasetname_list_y),
                     ('Predict', predict_list_y),
                     ('Atlas', atlas_list_y),
                     ('Connectivity', connectivity_list_y),
                     ('Model', name_regr_list_y),
                     ('Sampling', sampling_namer_list_y),
                     ('Confounds', confounds_name_list_y),
                     ('Regressout', do_conf_regressout_list_y),
                     ('Subj_number', subj_number_y),
                     ('Scan_number', scan_number_y),
                     ('y_true', pd.Series(y_test_array)),
                     ('y_predict', pd.Series(y_predict_array)),
                     ('ids_test', pd.Series(ids_test)))))

        if to_csv is True:
            name_file_save = (name_csv_prediction[0:-4] + '_' + name_regr
                              + '_' + str(subj_number_unique))
            save_timeseries_to_pkl(y_test_array, results_path,
                                   name_file=name_file_save,
                                   suffix='_ytrue',
                                   extension=None)
            save_timeseries_to_pkl(y_predict_array, results_path,
                                   name_file=name_file_save,
                                   suffix='_ypredict',
                                   extension=None)
            save_timeseries_to_pkl(ids_train, results_path,
                                   name_file=name_file_save,
                                   suffix='_ids_train',
                                   extension=None)
            save_timeseries_to_pkl(ids_test, results_path,
                                   name_file=name_file_save,
                                   suffix='_ids_test',
                                   extension=None)
            # save accuracy for each cv fold
            save_timeseries_to_pkl(dict_prediction_uni_cvfold, results_path,
                                   name_file=name_file_save,
                                   suffix='_cvfold',
                                   extension=None)
            print('The result csv file %s' % os.path.join(results_path,
                                                  name_file_save + '_cvfold'))
    datasetname_list = [dataset_name] * len(mse_std_list)
    predict_list = [predict_name] * len(mse_std_list)
    atlas_list = [atlas] * len(mse_std_list)
    connectivity_list = [connectivity_measure] * len(mse_std_list)

    sampling_namer_list = [sampling_name] * len(mse_std_list)
    confounds_name_list = [confounds_name] * len(mse_std_list)
    do_conf_regressout_list = [do_conf_regressout] * len(mse_std_list)

    subj_number_list = [subj_number_unique] * len(mse_std_list)
    scan_number_list = [subj_number_unique] * len(mse_std_list)

    df_prediction_uni = pd.DataFrame(
        OrderedDict((('Dataset', pd.Series(datasetname_list)),
                     ('Predict', pd.Series(predict_list)),
                     ('Atlas', pd.Series(atlas_list)),
                     ('Connectivity', pd.Series(connectivity_list)),
                     ('Sampling', pd.Series(sampling_namer_list)),
                     ('Confounds', pd.Series(confounds_name_list)),
                     ('Regressout', pd.Series(do_conf_regressout_list)),
                     ('Model', pd.Series(name_regr_list)),
                     ('Subj_number', pd.Series(subj_number_list)),
                     ('Scan_number', pd.Series(scan_number_list)),
                     ('MSE_test', pd.Series(mse_list)),
                     ('MAE_test', pd.Series(mae_list)),
                     ('EVS_test', pd.Series(evs_list)),
                     ('R2S_test', pd.Series(r2s_list)),
                     ('MSE_std_test', pd.Series(mse_std_list)),
                     ('MAE_std_test', pd.Series(mae_std_list)),
                     ('EVS_std_test', pd.Series(evs_std_list)),
                     ('R2S_std_test', pd.Series(r2s_std_list)),
                     ('MSE_train', pd.Series(mse_list_train)),
                     ('MAE_train', pd.Series(mae_list_train)),
                     ('EVS_train', pd.Series(evs_list_train)),
                     ('R2S_train', pd.Series(r2s_list_train)),
                     ('MSE_std_train', pd.Series(mse_std_list_train)),
                     ('MAE_std_train', pd.Series(mae_std_list_train)),
                     ('EVS_std_train', pd.Series(evs_std_list_train)),
                     ('R2S_std_train', pd.Series(r2s_std_list_train)))))

    if to_csv is True:
        df_prediction_uni.to_csv(os.path.join(results_path,
                                              name_csv_prediction))

        print('The result csv file %s' % os.path.join(results_path,
                                                  name_csv_prediction))
    return (df_prediction_uni, y_test_array, y_predict_array,
            df_y_test_y_prediction, regression_model_fit)






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
