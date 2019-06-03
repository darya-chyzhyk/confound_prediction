



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
