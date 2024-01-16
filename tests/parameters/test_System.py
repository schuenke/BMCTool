import pytest

from bmctool.parameters import System


@pytest.mark.parametrize(
    'b0, gamma, b0_inhom, rel_b1',
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
    'b0, gamma, b0_inhom, rel_b1',
    [
        ('three', 42.5764, 0.0, 1.0),  # b0 str (not convertable to float)
        (-3.0, 42.5764, 0.0, 1.0),  # b0 negative
        (3.0, 42.5764, 0.0, -1.0),  # rel_b1 negative
    ],
)
def test_raise_valueerror_for_invalid_values(b0, gamma, b0_inhom, rel_b1):
    """Test Options instantiation from invalid parameter types."""
    with pytest.raises(ValueError):
        System(b0=b0, gamma=gamma, b0_inhom=b0_inhom, rel_b1=rel_b1)
