
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from confound_prediction.data_simulation import simulate_confounded_data
from confound_prediction.sampling import (random_index_2remove,
                                         confound_isolating_index_2remove,
                                         confound_isolating_sampling,
                                         random_sampling)

from confound_prediction.deconfounding import (confound_isolating_cv,
                                              confound_regressout)



# Simulate data
X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=1000,
                          n_features=100)

cv_folds = 10
mi_rs_cv = []
corr_rs_cv = []
mi_ci_cv = []
corr_ci_cv = []


for cv_fold in range(cv_folds):
    print(cv_fold)
    # randome sampling
    ids_rs, mi_rs, corr_rs = random_sampling(y, z,
                                             min_sample_size=None,
                                             n_remove=None)
    mi_rs_cv.append(mi_rs)
    corr_rs_cv.append(corr_rs)


    # confound isolation
    ids_ci, mi_ci, corr_ci = confound_isolating_sampling(y, z, random_seed=0,
                                min_sample_size=None,
                                n_remove=None)

    mi_ci_cv.append(mi_ci)
    corr_ci_cv.append(corr_ci)



# TODO joblib option
#
# rs_cv = Parallel(n_jobs=5, verbose=1)(
#                 delayed(random_sampling)(y, z, min_sample_size=None,
#                                          n_remove=None)
#                 for cv_fold in range(cv_folds))
#
# ci_cv = Parallel(n_jobs=5, verbose=1)(
#                 delayed(confound_isolating_sampling)(y, z, random_seed=None,
#                                                      min_sample_size=None,
#                                                      n_remove=None)
#                 for cv_fold in range(cv_folds))
# for cv_fold in range(cv_folds):
#     ids_rs.append(rs_cv[]), mi_rs_cv, corr_rs_cv
# ids_rs, mi_ci_cv, corr_ci_cv




mi_rs_cv = np.array(mi_rs_cv)
corr_rs_cv = np.array(corr_rs_cv)
mi_ci_cv = np.array(mi_ci_cv)
corr_ci_cv = np.array(corr_ci_cv)

# Plotting Mutul Information

# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')


f, [ax1, ax2] = plt.subplots(1, 2, sharex=True, sharey=True,
                                           figsize=(7, 5))

ax1.plot(mi_rs_cv.T)
ax2.plot(mi_ci_cv.T)
ax1.set_title('Mutual, random sampling')
ax2.set_title('Mutual, Confound isoaltion')


# Plotting correlation


f, [ax1, ax2] = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7, 5))

ax1.plot(corr_rs_cv.T)
ax2.plot(corr_ci_cv.T)
ax1.set_title('Correlation, random sampling')
ax2.set_title('Correlation, Confound isoaltion')


