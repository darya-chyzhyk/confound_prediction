"""
Example for visualiation the evolution of the 'Confound isolation
cross-validation' and deconfounding methods.
In this example first we generate the data 'X', targets 'y' with direct link
with confounds 'z'.
Then we visualize the correlation and mutual information of each step of
sampling.

"""
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
                          n_features=10)

cv_folds = 2
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
    ids_ci, mi_ci, corr_ci = confound_isolating_sampling(y, z, random_seed=None,
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


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')


ax1.plot(mi_rs_cv.T)
ax2.plot(mi_ci_cv.T)
ax3.plot(corr_rs_cv.T)
ax4.plot(corr_ci_cv.T)

axes = [ax1, ax2, ax3, ax4]
for ax in axes:
    # Zero line
    ax.axhline(y=0.0, color='gray', linestyle='-')
    # Labels x-axes
    # ax.set_xlabel("Number of sampled subjects",fontsize=20)

# ax1.set_xlabel("Number of sampled subjects", fontsize=16)
# ax3.set_xlabel("Number of sampled subjects", fontsize=16)
# ax3.yaxis.label.set_position((0, 0.1))
# ax3.set_ylabel("Number of sampled subjects", fontsize=16)

# plt.xlabel("Number of sampled subjects", fontsize=16)

f.text(0.5, 0.1, 'Number of sampled subjects', va='center', ha='center',
       fontsize=16)
# Axes

ax1.set_ylabel('Mutual\nInformation', fontsize=16)
ax3.set_ylabel('Correlation', fontsize=16)


# Titels

ax1.set_title('Random sampling', fontsize=16)
ax2.set_title('Confound isoaltion cv', fontsize=16)
# ax3.set_title('Random sampling', fontsize=16)
# ax4.set_title('Confound isoaltion cv', fontsize=16)

f.tight_layout()




#
#
# if score == 'mutual':
#     ax.set_ylabel('Mutual Information', fontsize=30)
# elif score == 'correlation':
#     ax.set_ylabel('Correlation between\ntarget and confound', fontsize=26)
# ax.set_xlabel("Number of sampled subjects",fontsize=30)
#
#
# ax1.set_title('Random sampling')
# ax2.set_title('Confound isoaltion cv')
# ax3.set_title('Random sampling')
# ax4.set_title('Confound isoaltion cv')
#
# f.tight_layout()



# Plotting correlation


# f, [ax1, ax2] = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7, 5))

# f, [ax1, ax2] = plt.subplots(1, 2, sharex=True, sharey=True,
#                                            figsize=(7, 5))
