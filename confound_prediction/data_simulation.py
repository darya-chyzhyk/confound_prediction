"""Generate the data
X - observation
y - target
z- confound
with 3 different properties:

1) No direct link between X and y
2) Direct link between X and y
3) Weak link between X and y"""

import numpy as np

def simulate_confounded_data(link_type='direct_link', n_samples=100, n_features=100):
    """

    :param link_type: str,
        Type of the links between target and confound. Options: "no_link",
        "direct_link", "weak_link"
    :param n_samples: int,
        number of samples
    :param n_features: int,
        number of features
    :return:
    """
    np.random.seed(42)

    mu, sigma = 0, 1.0  # mean and standard deviation
    x_rand = np.random.normal(mu, sigma, [n_samples, n_features])
    y_rand = np.random.normal(mu, sigma, n_samples)
    z_rand = np.random.normal(mu, sigma, n_samples)

    if link_type == 'no_link':
        y = np.copy(y_rand)
        z = 1 * y_rand + z_rand
        X = x_rand + z.reshape(-1, 1)
    elif link_type == 'direct_link':
        y = np.copy(y_rand)
        z = y_rand + z_rand
        X = x_rand + y_rand.reshape(-1, 1) + z.reshape(-1, 1)
    elif link_type == 'weak_link':
        y = np.copy(y_rand)
        z = 0.5 * y_rand + z_rand
        X = x_rand + y_rand.reshape(-1, 1) + z.reshape(-1, 1)
    return X, y, z
