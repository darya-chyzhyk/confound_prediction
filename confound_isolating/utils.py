import numpy as np


def _ensure_int_positive(data, default=None):
    """
    Function that ensure that data is positive and integer.
    If data is None change it to default value.
    If the data is float, convert it to int.
    :param data:
    :param default:
    :return:
    """

    if data is None:
        if default is None:
            raise TypeError(str(data) + ' or ' + default + " must be positive "
                                                     "integer")
        else:
            data = default
    else:
        if isinstance(data, (list, tuple, str, np.ndarray)) or data < 0:
            raise TypeError(str(data) + " keyword has an unhandled type: %s"
                            % data.__class__ + " or it is not positive")

        if isinstance(data, float):
            data = int(data)
    return data

