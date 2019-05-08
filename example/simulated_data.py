

from confound_isolating.data_simulation import simulate_confounded_data

# Simulate data
X, y, z, = simulate_confounded_data(link_type='direct_link', n_samples=100,
                          n_features=100)

ids = list(range(0, y.shape[0]))


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
