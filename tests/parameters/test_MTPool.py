from copy import deepcopy

import pytest
from bmctool.parameters import MTPool


@pytest.mark.parametrize(
    ('r1', 'r2', 'k', 'f', 'dw', 't1', 't2', 'lineshape'),
    [
        (1.0, 2.0, 3.0, 0.5, 5.0, None, None, 'lorentzian'),  # all values float
        (1.0, 2.0, 3.0, 0.5, 5, None, None, 'lorentzian'),  # dw int
        (1.0, 2.0, 3.0, 0.5, '5', None, None, 'lorentzian'),  # dw str
        (None, 2.0, 3.0, 0.5, 5.0, 1.0, None, 'lorentzian'),  # r1 None, but t1 float
        (None, 2.0, 3.0, 0.5, 5.0, '1.0', None, 'lorentzian'),  # r1 None, but t1 str
        (1.0, None, 3.0, 0.5, 5.0, None, 2.0, 'superlorentzian'),  # r2 None, but t2 float
        (1.0, 2, '3.0', '1e-1', 5.0, None, None, 'superlorentzian'),  # k str, f str (scientific notation)
    ],
)
def test_from_valid_params(r1, r2, k, f, dw, t1, t2, lineshape):
    """Test CESTPool instantiation from valid parameters."""
    a = MTPool(r1=r1, r2=r2, k=k, f=f, dw=dw, t1=t1, t2=t2, lineshape=lineshape)

    # assert that the attributes are set correctly
    assert a.r1 == float(r1) if r1 is not None else 1 / float(t1)
    assert a.r2 == float(r2) if r2 is not None else 1 / float(t2)
    assert a.k == float(k)
    assert a.f == float(f)
    assert a.dw == float(dw)
    assert a.lineshape == lineshape


@pytest.mark.parametrize(
    ('r1', 'r2', 'k', 'f', 'dw', 't1', 't2', 'lineshape'),
    [
        (1.0, 2.0, 3.0, 4.0, 5.0, None, None, 'lorentzian'),  # f not between 0 and 1
        (-1.0, 2.0, 3.0, 0.5, 5.0, None, None, 'lorentzian'),  # r1 negative
        (None, 2.0, 3.0, 4.0, 5.0, -1.0, None, 'superlorentzian'),  # t1 negative
        (None, 2.0, -3.0, 4.0, 5.0, -1.0, None, 'superlorentzian'),  # k negative
        (1.0, 2.0, 3.0, 1.0, 5.0, None, None, 'other_shape'),  # invalid lineshape
    ],
)
def test_from_invalid_values(r1, r2, k, f, dw, t1, t2, lineshape):
    """Test CESTPool instantiation from invalid parameter types."""
    with pytest.raises(ValueError):
        MTPool(r1=r1, r2=r2, k=k, f=f, dw=dw, t1=t1, t2=t2, lineshape=lineshape)


def test_slots_property():
    """Test that __slots__ is properly set."""
    a = MTPool(r1=1.0, r2=2.0, k=3.0, f=0.5, dw=5.0, lineshape='lorentzian')
    with pytest.raises(AttributeError):
        a.new_attr = 1.0  # type: ignore


def test_from_dict_classmethod():
    """Test MTPool instantiation from a dictionary."""
    d = {'r1': 1.0, 'r2': 2.0, 'k': 3.0, 'f': 0.5, 'dw': 5.0, 'lineshape': 'lorentzian'}
    a = MTPool.from_dict(d)
    b = MTPool(**d)  # type: ignore
    assert a == b


def test_k_value_setter():
    """Test that k value is set correctly."""
    a = MTPool(r1=1.0, r2=2.0, k=3.0, f=0.5, dw=5.0, lineshape='lorentzian')
    assert a.k == 3.0
    a.k = 4.0
    assert a.k == 4.0
    with pytest.raises(ValueError):
        a.k = -4.0


def test_equality(valid_mt_pool_object):
    """Test that MTPool instances are equal if their attributes are equal."""
    a = deepcopy(valid_mt_pool_object)
    b = MTPool(r1=1.0, r2=2.0, k=3.0, f=0.5, dw=5.0, lineshape='lorentzian')
    c = MTPool(r1=2.0, r2=2.0, k=3.0, f=0.5, dw=5.0, lineshape='lorentzian')
    d = MTPool(r1=1.0, r2=1.0, k=3.0, f=0.5, dw=5.0, lineshape='lorentzian')
    e = MTPool(r1=1.0, r2=2.0, k=4.0, f=0.5, dw=5.0, lineshape='lorentzian')
    f = MTPool(r1=1.0, r2=2.0, k=3.0, f=0.6, dw=5.0, lineshape='lorentzian')
    g = MTPool(r1=1.0, r2=2.0, k=3.0, f=0.5, dw=6.0, lineshape='lorentzian')
    h = MTPool(r1=1.0, r2=2.0, k=3.0, f=0.5, dw=5.0, lineshape='superlorentzian')
    i = 'not a MTPool object'

    assert a == valid_mt_pool_object
    assert a == b
    assert a != c
    assert a != d
    assert a != e
    assert a != f
    assert a != g
    assert a != h
    assert a != i
