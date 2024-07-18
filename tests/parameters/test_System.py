from copy import deepcopy

import pytest
from bmctool.parameters import System


@pytest.mark.parametrize(
    ('b0', 'gamma', 'b0_inhom', 'rel_b1'),
    [
        (3.0, 42.5764, 0.0, 1.0),  # all values correct type
        (3, 42.5764, 0, 1),  # b0, b0_inhom, rel_b1 int instead of float
        ('3.0', '42.5764', '0', '1.0'),  # all values str instead of float
    ],
)
def test_from_valid_params(b0, gamma, b0_inhom, rel_b1):
    """Test Options instantiation from valid parameters."""
    a = System(b0=b0, gamma=gamma, b0_inhom=b0_inhom, rel_b1=rel_b1)

    # assert that the attributes are set correctly
    assert a.b0 == float(b0)
    assert a.gamma == float(gamma)
    assert a.b0_inhom == float(b0_inhom)
    assert a.rel_b1 == float(rel_b1)


@pytest.mark.parametrize(
    ('b0', 'gamma', 'b0_inhom', 'rel_b1'),
    [
        ('three', 42.5764, 0.0, 1.0),  # b0 str (not convertible to float)
        (-3.0, 42.5764, 0.0, 1.0),  # b0 negative
        (3.0, 42.5764, 0.0, -1.0),  # rel_b1 negative
    ],
)
def test_raise_valueerror_for_invalid_values(b0, gamma, b0_inhom, rel_b1):
    """Test Options instantiation from invalid parameter types."""
    with pytest.raises(ValueError):
        System(b0=b0, gamma=gamma, b0_inhom=b0_inhom, rel_b1=rel_b1)


def test_equality(valid_system_object):
    """Test that System instances are equal if their attributes are equal."""
    a = deepcopy(valid_system_object)
    b = System(b0=3.0, gamma=42.5764, b0_inhom=0.0, rel_b1=1.0)
    c = System(b0=7.0, gamma=42.5764, b0_inhom=0.0, rel_b1=1.0)
    d = System(b0=3.0, gamma=42.5678, b0_inhom=0.0, rel_b1=1.0)
    e = System(b0=3.0, gamma=42.5764, b0_inhom=0.1, rel_b1=1.0)
    f = System(b0=3.0, gamma=42.5764, b0_inhom=0.0, rel_b1=1.1)
    g = 'not a System object'

    assert a == valid_system_object
    assert a == b
    assert a != c
    assert a != d
    assert a != e
    assert a != f
    assert a != g
