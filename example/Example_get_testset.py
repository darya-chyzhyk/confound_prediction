'''
Simple example, where we
1) generate the data
2) create the test and train sets; in the test set 'y' and 'z' are independent
'''

from confound_prediction.data_simulation import simulate_confounded_data

from confound_prediction.deconfounding import (confound_isolating_cv,
                                              confound_regressout)


# Simulate data
X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=100,
                          n_features=2)

# Create train and test sets with "Confound-isolationg cross-validation"
x_test_cicv, x_train_cicv, y_test_cicv, y_train_cicv, \
ids_test_cicv, ids_train_cicv = \
    confound_isolating_cv(X, y, z, random_seed=None, min_sample_size=None,
                          cv_folds=10, n_remove=None)


# Create train and test sets, regressing out the confounds out-of-samle
x_test_oos, x_train_oos, y_test_oos, y_train_oos, ids_test_oos, ids_train_oos \
    = confound_regressout(X, y, z, type_deconfound='out_of_sample',
                          min_sample_size=None, cv_folds=10, n_remove=None)



