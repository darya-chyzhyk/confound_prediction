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

# Create train and test sets
x_test, x_train, y_test, y_train, ids_test, ids_train = \
    confound_regressout(X, y, z, type_deconfound='out_of_sample',
                        min_sample_size=10.2,
                        cv_folds=10, n_remove=10)



