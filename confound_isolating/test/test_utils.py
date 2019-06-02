# TODO test _ensure_int_positive(data, default=None)

from confound_isolating.utils import _ensure_int_positive

from nose.tools import assert_raises


def test_ensure_int_positive():
    assert _ensure_int_positive(5, default=None) == 5
    assert _ensure_int_positive(5.1, default=None) == 5
    assert _ensure_int_positive(5, default=10) == 5
    assert _ensure_int_positive(None, default=5) == 5

    assert_raises(TypeError, _ensure_int_positive, -5.1, None)
    assert_raises(TypeError, _ensure_int_positive, None, None)



