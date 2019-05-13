

from confound_isolating.data_simulation import simulate_confounded_data

# Simulate data
X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=100,
                          n_features=100)
ids = list(range(0, y.shape[0]))

# Pre-deconfounding if neccessery

if do_conf_regressout is 'jointly':
    X = clean(X, standardize=False, detrend=False,
                    confounds=z, low_pass=None, high_pass=None)




# Sampling


# Deconfound and create test and train sets

# Prediction

if (n_jobs == 0) or (n_jobs is None):
    sampled_list = []
    for n_seed in range(n_seeds):
        sampled_list.append(sampling(x_sampling, y_sampling, ids,
                                     type_sampling=type_sampling, n_seed=n_seed,
                                     min_size=min_size,
                                     save_mi_iter=save_mi_iter))
else:
    sampled_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(sampling)(x_sampling, y_sampling, ids,
                          type_sampling=type_sampling, n_seed=n_seed,
                          min_size=min_size, save_mi_iter=save_mi_iter)
        for n_seed in range(n_seeds))


# clear signal



def _standardize(signals, detrend=False, normalize=True):
    """ Center and norm a given signal (time is along first axis)

    Parameters
    ----------
    signals: numpy.ndarray
        Timeseries to standardize

    detrend: bool
        if detrending of timeseries is requested

    normalize: bool
        if True, shift timeseries to zero mean value and scale
        to unit energy (sum of squares).

    Returns
    -------
    std_signals: numpy.ndarray
        copy of signals, normalized.
    """



    if normalize:
        if signals.shape[0] == 1:
            warnings.warn('Standardization of 3D signal has been requested but '
                'would lead to zero values. Skipping.')
            return signals

        if not detrend:
            # remove mean if not already detrended
            signals = signals - signals.mean(axis=0)

        std = np.sqrt((signals ** 2).sum(axis=0))
        std[std < np.finfo(np.float).eps] = 1.  # avoid numerical problems
        signals /= std
    return signals




def clean_conf(confounds):
    confounds = _ensure_float(confounds)

    # Apply low- and high-pass filters to keep filters orthogonal
    # (according to Lindquist et al. (2018))


    confounds = _standardize(confounds, normalize=standardize,
                             detrend=detrend)

    if not standardize:
        # Improve numerical stability by controlling the range of
        # confounds. We don't rely on _standardize as it removes any
        # constant contribution to confounds.
        confound_max = np.max(np.abs(confounds), axis=0)
        confound_max[confound_max == 0] = 1
        confounds /= confound_max

    # Pivoting in qr decomposition was added in scipy 0.10
    Q, R, _ = linalg.qr(confounds, mode='economic', pivoting=True)
    Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100.]
    signals -= Q.dot(Q.T).dot(signals)
