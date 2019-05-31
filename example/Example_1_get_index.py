

from confound_isolating.data_simulation import simulate_confounded_data
from confound_isolating.sampling import (random_index_2remove,
                                         confound_isolating_index_2remove,
                                         confound_isolating_sampling,
                                         random_sampling)

from confound_isolating.deconfounding import (confound_isolating_cv,
                                              confound_regressout)



# Simulate data
X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=100,
                          n_features=2)

# x_test, x_train, y_test, y_train, ids_test, ids_train = \
#     confound_isolating_cv(X, y, z, random_seed=0, min_sample_size=None,
#                           cv_folds=10, type_bandwidth='scott')

x_test, x_train, y_test, y_train, ids_test, ids_train = \
    confound_isolating_cv(X, y, z, random_seed=0, min_sample_size=None,
                          cv_folds=10, n_remove=None)



# x_test, x_train, y_test, y_train, ids_test, ids_train = \
#     confound_regressout(X, y, z, type_deconfound='out_of_sample',
#                         min_sample_size=None,
#                         cv_folds=10, n_remove=10)







# a, b, c = random_sampling(y, z, min_sample_size=30, type_bandwidth='2scott')
#
# ind_ci, mi_ci, cor_ci = confound_isolating_sampling(y, z, n_seed=0,
#                                             min_sample_size=None,
#                                 type_bandwidth='scott')


# ids = list(range(0, y.shape[0]))

# Remove index with random or Confound Izolating methods


# ratio_dens_1, index_test_remove_1, kde_y_1, kde_z_1, kde_yz_1, scale_method = \
#     random_index_2remove(y, z)
#
# ratio_dens, ratio_remove, index_test_remove, kde_y, kde_z, kde_yz = \
#     confound_isolating_index_2remove(y, z)




# a, b, c = random_sampling(y, z, min_sample_size=None, type_bandwidth='scott')


