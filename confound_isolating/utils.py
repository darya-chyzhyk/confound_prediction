import numpy as np

def _ensure_int_positive(var, default=None):
    """
    Function that ensure that parameter is positive and integer and if it is
    None change it to default value
    :param var:
    :param default:
    :return:
    """


    if var is None:
        if default is None:
            raise TypeError(var + ' or ' + default + " must be positive "
                                                     "integer")
        else:
            var = default

    else:
        if isinstance(var, (list, tuple, str, np.ndarray)) or var < 0:
            raise TypeError(var + " keyword has an unhandled type: %s"
                            % var.__class__ + " or it is not positive")

        if isinstance(var, float):
            var = int(var)
    return var
