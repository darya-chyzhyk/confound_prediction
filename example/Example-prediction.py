"""
Example of prediction of 'y' from 'X' with presence of confound 'z' (direct
link between 'y' and 'z') with 4 different deconfound strategies:
1. Confound Isolation cross-validation method
2. 'Out_of_sample' deconfounding
3. 'Jointly' deconfounding
4. Without deconfounding

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
X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=100,
                                    n_features=100)
print('Simulated data contains ', X.shape[0], ' - samples and ', X.shape[1],
      ' - features')

# Get the train and test data with Confound Isolation cross-validation method
print('Confound Isolation cross-validation method is processing.....')
x_test_cicv, x_train_cicv, y_test_cicv, y_train_cicv, _, _ = \
    confound_isolating_cv(X, y, z, random_seed=None, min_sample_size=None,
                          cv_folds=10, n_remove=None)

# Get the train and test data with 'out_of_sample' deconfounding
print('Out_of_sample deconfounding method is processing.....')
x_test_oos, x_train_oos, y_test_oos, y_train_oos, _, _ = \
    confound_regressout(X, y, z, type_deconfound='out_of_sample',
                        min_sample_size=None, cv_folds=10, n_remove=None)

# Get the train and test data without deconfounding
print('Without deconfounding .....')
x_test_fa, x_train_fa, y_test_fa, y_train_fa, _, _ = \
    confound_regressout(X, y, z, type_deconfound='False',
                        min_sample_size=None, cv_folds=10, n_remove=None)

# Get the train and test data with 'jointly' deconfounding
print('Deconfound jointly .....')
x_test_jo, x_train_jo, y_test_jo, y_train_jo, _, _ = \
    confound_regressout(X, y, z, type_deconfound='jointly',
                        min_sample_size=None, cv_folds=10, n_remove=None)

# Prediction
model = RidgeCV()

mse_cicv, mae_cicv, evs_cicv, r2s_cicv = \
    model_fit_datasplit(x_test_cicv, x_train_cicv, y_test_cicv, y_train_cicv,
                        model)

mse_oos, mae_oos, evs_oos, r2s_oos = \
    model_fit_datasplit(x_test_oos, x_train_oos, y_test_oos, y_train_oos,
                        model)

mse_jo, mae_jo, evs_jo, r2s_jo = \
    model_fit_datasplit(x_test_jo, x_train_jo, y_test_jo, y_train_jo, model)

mse_fa, mae_fa, evs_fa, r2s_fa = model_fit_datasplit(x_test_fa, x_train_fa,
                                                     y_test_fa, y_train_fa,
                                                     model)
mae_plot = [np.array(mae_cicv), np.array(mae_oos), np.array(mae_jo),
            np.array(mae_fa)]

r2s_plot = [np.array(r2s_cicv), np.array(r2s_oos), np.array(r2s_jo),
            np.array(r2s_fa)]

df_mae = pd.DataFrame({'cicv': mae_cicv,
                       'oos': mae_oos,
                       'ma': mae_jo,
                       'fa': mae_fa})
df_mae_plot = pd.melt(df_mae.reset_index(),
                      value_vars=df_mae.columns.values.tolist(),
                      var_name='confound', value_name='value')

df_r2s = pd.DataFrame({'cicv': r2s_cicv,
                       'oos': r2s_oos,
                       'ma': r2s_jo,
                       'fa': r2s_fa})
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

labels = ['Confound \n isolation cv',
          'Out-of-sample \n deconfounding',
          'Deconfounding \n test and train\njointly',
          'Without \n deconfounding']

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