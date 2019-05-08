"""Generate the data
X - observation
y - target
z- confound
with 3 different properties:

1) No direct link between y and z
2) Direct link between y and z
3) Weak link between y"""

import numpy as np

def simulate_confounded_data(link_type, n_samples=100, n_features=100):
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

    if exper == '1':
        y = np.copy(y_rand)
        z_conf = 1 * y_rand + z_rand
        X_brain = x_rand + z_conf.reshape(-1, 1)
    elif exper == '2':
        y = np.copy(y_rand)
        z_conf = 0.5 * y_rand + z_rand
        X_brain = x_rand + z_conf.reshape(-1, 1)

    elif exper == '3':
        y = np.copy(y_rand)
        z_conf = 0.1 * y_rand + z_rand
        X_brain = x_rand + z_conf.reshape(-1, 1)

elif exper == '4':
    y = np.copy(y_rand)
    z_conf = 1 * y_rand ** 2 + z_rand
    X_brain = x_rand + z_conf.reshape(-1, 1)

elif exper == '5':
    y = np.copy(y_rand)
    z_conf = 0.5 * y_rand ** 2 + z_rand
    X_brain = x_rand + z_conf.reshape(-1, 1)

elif exper == '6':
    y = np.copy(y_rand)
    z_conf = 0.1 * y_rand ** 2 + z_rand
    X_brain = x_rand + z_conf.reshape(-1, 1)

elif exper == '7':
    y = np.copy(y_rand)
    z_conf = 1 * y_rand + z_rand
    X_brain = x_rand + y.reshape(-1, 1) + z_conf.reshape(-1, 1)
elif exper == '8':
    y = np.copy(y_rand)
    z_conf = 0.5 * y_rand + z_rand
    X_brain = x_rand + y.reshape(-1, 1) + z_conf.reshape(-1, 1)

elif exper == '9':
    y = np.copy(y_rand)
    z_conf = 0.1 * y_rand + z_rand
    X_brain = x_rand + y.reshape(-1, 1) + z_conf.reshape(-1, 1)

elif exper == '10':
    y = np.copy(y_rand)
    z_conf = 1 * y_rand ** 2 + z_rand
    X_brain = x_rand + y.reshape(-1, 1) + z_conf.reshape(-1, 1)

elif exper == '11':
    y = np.copy(y_rand)
    z_conf = 0.5 * y_rand ** 2 + z_rand
    X_brain = x_rand + y.reshape(-1, 1) + z_conf.reshape(-1, 1)

elif exper == '12':
    y = np.copy(y_rand)
    z_conf = 0.1 * y_rand ** 2 + z_rand
    X_brain = x_rand + y.reshape(-1, 1) + z_conf.reshape(-1, 1)
