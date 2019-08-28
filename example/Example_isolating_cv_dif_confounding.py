"""
Example of prediction of 'y' from 'X' with presence of confound 'z':
1. "direct link"
2. "weak_link"
3. "no_link"
The link is between 'y' and 'z'.
We are using "Confound Isolation cross-validation method.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from confound_prediction.data_simulation import simulate_confounded_data

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
        # print('Start prediction with ', model)
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
    return mse, mae, evs, r2s


# Simulate data
X_direct, y_direct, z_direct = simulate_confounded_data(
    link_type='direct_link', n_samples=100, n_features=1000)

X_weak, y_weak, z_weak = simulate_confounded_data(link_type='weak_link',
                                                  n_samples=100,
                                                  n_features=1000)

X_no, y_no, z_no = simulate_confounded_data(link_type='no_link',
                                            n_samples=100, n_features=1000)

print('Confound Isolation cross-validation method is processing.....')
x_test_direct, x_train_direct, y_test_direct, y_train_direct, _, _ = \
    confound_isolating_cv(X_direct, y_direct, z_direct, cv_folds=10)

x_test_weak, x_train_weak, y_test_weak, y_train_weak, _, _ = \
    confound_isolating_cv(X_weak, y_weak, z_weak, cv_folds=10)

x_test_no, x_train_no, y_test_no, y_train_no, _, _ = \
    confound_isolating_cv(X_no, y_no, z_no, cv_folds=10)

# Prediction
model = RidgeCV()

mse_direct, mae_direct, evs_direct, r2s_direct = \
    model_fit_datasplit(x_test_direct, x_train_direct, y_test_direct,
                        y_train_direct, model)

mse_weak, mae_weak, evs_weak, r2s_weak = \
    model_fit_datasplit(x_test_weak, x_train_weak, y_test_weak, y_train_weak,
                        model)

mse_no, mae_no, evs_no, r2s_no = model_fit_datasplit(x_test_no, x_train_no,
                                                     y_test_no, y_train_no,
                                                     model)


#######
mae_plot = [np.array(mae_direct), np.array(mae_weak), np.array(mae_no)]

r2s_plot = [np.array(r2s_direct), np.array(r2s_weak), np.array(r2s_no)]

df_mae = pd.DataFrame({'direct': mae_direct,
                       'weak': mae_weak,
                       'no': mae_no})
df_mae_plot = pd.melt(df_mae.reset_index(),
                      value_vars=df_mae.columns.values.tolist(),
                      var_name='confound', value_name='value')

df_r2s = pd.DataFrame({'direct': r2s_direct,
                       'weak': r2s_weak,
                       'no': r2s_no})
df_r2s_plot = pd.melt(df_r2s.reset_index(),
                      value_vars=df_r2s.columns.values.tolist(),
                      var_name='confound', value_name='value')

# Plotting
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor': 'white'})
for i in range(1, 16, 2):
    ax1.axvspan(i-0.5, i+0.5, facecolor='white', zorder=-1)
    ax2.axvspan(i - 0.5, i + 0.5, facecolor='white', zorder=-1)

# MAE
sns.boxplot(x="confound", y="value", data=df_mae_plot, palette="colorblind",
            ax=ax1)

sns.stripplot(x="confound", y="value", data=df_mae_plot, jitter=True,
              dodge=True, marker='o', alpha=0.7, size=12, edgecolor='black',
              linewidth=1.5, palette="colorblind", ax=ax1)

# R2s
sns.boxplot(x="confound", y="value", data=df_r2s_plot, palette="colorblind",
            ax=ax2)

sns.stripplot(x="confound", y="value", data=df_r2s_plot, jitter=True,
              dodge=True, marker='o', alpha=0.7, size=12, edgecolor='black',
              linewidth=1.5, palette="colorblind", ax=ax2)

# Tickes
ax1.axhline(y=0.0, color='black', linestyle='-')
ax2.axhline(y=0.0, color='black', linestyle='-')

labels = ['Strong link',
          'Weak link',
          'No link']

ax1.set_xticklabels(labels, fontsize=16, rotation=70)
ax2.set_xticklabels(labels, fontsize=16, rotation=70)
ax1.xaxis.set_tick_params(length=5)
ax2.xaxis.set_tick_params(length=5)
ax1.yaxis.set_tick_params(labelsize=14, length=5)
ax2.yaxis.set_tick_params(labelsize=14, length=5)

# Axes
ax1.set_title('Mean absolute error', fontsize=24)
ax2.set_title(r'$R^2  score$', fontsize=24)

ax1.set_ylabel("Mean absolute error",fontsize=16)
ax2.set_ylabel("R2S score",fontsize=16)
ax1.set_xlabel("",fontsize=30)
ax2.set_xlabel("",fontsize=30)

plt.gcf().subplots_adjust(bottom=0.4, left=0.1, right=0.95, wspace=0.3)
