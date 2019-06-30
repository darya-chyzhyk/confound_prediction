"""
Example for visualization the evolution of the 'Confound isolation
cross-validation' and deconfounding methods.
In this example first we generate the data 'X', targets 'y' with direct link
with confounds 'z'.
Then we visualize the correlation and mutual information of each step of
sampling.

"""
import numpy as np
import matplotlib.pyplot as plt

from confound_prediction.data_simulation import simulate_confounded_data
from confound_prediction.sampling import (confound_isolating_sampling,
                                          random_sampling)


def list_to_array(x):
    x_array = np.zeros([len(x), len(max(x, key = lambda a: len(a)))])
    for i,j in enumerate(x):
        x_array[i][0:len(j)] = j
    return x_array


# Simulate data
X, y, z = simulate_confounded_data(link_type='direct_link', n_samples=1000,
                                   n_features=100)
# Define parameters
cv_folds = 10

mi_rs_cv = []
corr_rs_cv = []
mi_ci_cv = []
corr_ci_cv = []

for cv_fold in range(cv_folds):
    print(cv_fold)
    # random sampling
    ids_rs, mi_rs, corr_rs = random_sampling(y, z, min_sample_size=None,
                                             n_remove=None)
    mi_rs_cv.append(mi_rs)
    corr_rs_cv.append(corr_rs)

    # confound isolation
    ids_ci, mi_ci, corr_ci = confound_isolating_sampling(y, z, random_seed=None,
                                min_sample_size=None, n_remove=None)
    mi_ci_cv.append(mi_ci)
    corr_ci_cv.append(corr_ci)

# Convert lists of list of unequal lengths to numpy array

mi_rs_array = np.array(mi_rs_cv)
corr_rs_array = np.array(corr_rs_cv)
mi_ci_array = list_to_array(mi_ci_cv)
corr_ci_array = list_to_array(corr_ci_cv)

# Plotting Mutual Information and Correlation
# Different colors represent different cv_fold

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

ax1.plot(mi_rs_array.T)
ax2.plot(mi_ci_array.T)
ax3.plot(corr_rs_array.T)
ax4.plot(corr_ci_array.T)

axes = [ax1, ax2, ax3, ax4]
for ax in axes:
    # Zero line
    ax.axhline(y=0.0, color='gray', linestyle='-')

# Axes
ax1.set_ylabel('Mutual\nInformation', fontsize=16)
ax3.set_ylabel('Correlation', fontsize=16)

# Titles
ax1.set_title('Random sampling', fontsize=16)
ax2.set_title('Confound isolation cv', fontsize=16)

f.text(0.5, 0.04, 'Number of sampled subjects', ha="center", va="center",
       fontsize=16)

plt.gcf().subplots_adjust(bottom=0.15, left=0.15, right=0.97)
plt.show()
