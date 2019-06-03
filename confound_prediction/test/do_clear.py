from nilearn.signal import clean

from confound_prediction.deconfounding import deconfound_jointly
from confound_prediction.data_simulation import simulate_confounded_data


X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=100,
                          n_features=100)

x_nilearn = clean(X, sessions=None, detrend=False, standardize=False,
                  confounds=z, low_pass=None, high_pass=None, t_r=None,
                  ensure_finite=False)

x_mio = deconfound_jointly(X, z)

print(x_mio==x_nilearn)
