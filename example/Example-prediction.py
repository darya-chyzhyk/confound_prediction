"""
Example of prediction
"""

import matplotlib.pyplot as plt
import numpy as np

from confound_prediction.data_simulation import simulate_confounded_data
from confound_prediction.sampling import (random_index_2remove,
                                         confound_isolating_index_2remove,
                                         confound_isolating_sampling,
                                         random_sampling)

from confound_prediction.deconfounding import (confound_isolating_cv,
                                               confound_regressout)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             explained_variance_score, r2_score)


def model_fit_datasplit(x_train_cv, x_test_cv, y_train_cv, y_test_cv, model):
    mse = []
    mae = []
    evs = []
    r2s = []
    for x_train, x_test, y_train, y_test in zip(x_train_cv, x_test_cv,
                                                y_train_cv, y_test_cv):
        print('Start prediction with ', model)


        model.fit(x_train, y_train)
        test_predict = model.predict(x_test)

        # Mean squared error
        mse.append(mean_squared_error(y_test, test_predict))
        # Mean absolute error
        mae.append(mean_absolute_error(y_test, test_predict))
        # Explained variance score
        evs.append(explained_variance_score(y_test, test_predict))
        # R^2 score
        r2s.append(r2_score(y_test, test_predict))
    return (mse, mae, evs, r2s)


# Simulate data
X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=100,
                                    n_features=2)

# Get the train and test data with Confound Isolation cross-validation method
x_test_cicv, x_train_cicv, y_test_cicv, y_train_cicv, _, _ = \
    confound_isolating_cv(X, y, z, random_seed=0, min_sample_size=None,
                          cv_folds=10, n_remove=None)

# Get the train and test data with 'out_of_sample' deconfounding
x_test_oos, x_train_oos, y_test_oos, y_train_oos, _, _ = \
    confound_regressout(X, y, z, type_deconfound='out_of_sample',
                        min_sample_size=None, cv_folds=10, n_remove=None)

# Get the train and test data with 'model_agnostic' deconfounding
x_test_ma, x_train_ma, y_test_ma, y_train_ma, _, _ = \
    confound_regressout(X, y, z, type_deconfound='model_agnostic',
                        min_sample_size=None, cv_folds=10, n_remove=None)

# Get the train and test data without deconfounding
x_test_fa, x_train_fa, y_test_fa, y_train_fa, _, _ = \
    confound_regressout(X, y, z, type_deconfound='False',
                        min_sample_size=None, cv_folds=10, n_remove=None)

# Prediction
model = RidgeCV()

mse_cicv, mae_cicv, evs_cicv, r2s_cicv = \
    model_fit_datasplit(x_test_cicv, x_train_cicv, y_test_cicv, y_train_cicv,
                        model)

mse_oos, mae_oos, evs_oos, r2s_oos = \
    model_fit_datasplit(x_test_oos, x_train_oos, y_test_oos, y_train_oos, model)

mse_ma, mae_ma, evs_ma, r2s_ma = \
    model_fit_datasplit(x_test_ma, x_train_ma, y_test_ma, y_train_ma, model)

mse_fa, mae_fa, evs_fa, r2s_fa = model_fit_datasplit(x_test_fa, x_train_fa, y_test_fa,
                                         y_train_fa, model)


mae_plot = [np.array(mae_cicv), np.array(mae_oos), np.array(mae_ma),
            np.array(mae_fa)]

r2s_plot = [np.array(r2s_cicv), np.array(r2s_oos), np.array(r2s_ma),
            np.array(r2s_fa)]

# Plotting




labels = ['Confound isolation', 'Out-of-sample', 'Model-agnostic',
          'Without deconfounding']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# rectangular box plot
bplot1 = axes[0].boxplot(mae_plot,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
axes[0].set_title('Rectangular box plot')

# notch shape box plot
bplot2 = axes[1].boxplot(r2s_plot,
                         notch=True,  # notch shape
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
axes[1].set_title('Comparison of prediction')

# fill with colors
colors = ['firebrick', 'olive', 'orange', 'steelblue']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Observed values')

plt.show()







