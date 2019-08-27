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
    link_type='direct_link', n_samples=100, n_features=100)
print('Simulated data contains ', X.shape[0], ' - samples and ', X.shape[1],
      ' - features')