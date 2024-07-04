import pytest
from bmctool.parameters.WaterPool import WaterPool


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 't1', 't2'),
    [
        (1.0, 2.0, 1.0, None, None),  # all values float
        (1.0, 2.0, 1, None, None),  # f int
        (1.0, 2.0, '1', None, None),  # f str
        (1.0, 2.0, '1e-0', None, None),  # f str (scientific notation)
        (None, 2.0, 1.0, 1.0, None),  # r1 None, but t1 float
        (None, 2.0, 1.0, '1.0', None),  # r1 None, but t1 str
        (1.0, None, 1.0, None, 0.5),  # r2 None, but t2 float
    ],
)
def test_from_valid_params(r1, r2, f, t1, t2):
    """Test WaterPool instantiation from valid parameters."""
    a = WaterPool(r1=r1, r2=r2, f=f, t1=t1, t2=t2)

    # assert that the attributes are set correctly
    assert a.r1 == float(r1) if r1 is not None else 1 / float(t1)
    assert a.r2 == float(r2) if r2 is not None else 1 / float(t2)
    assert a.f == float(f)


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 't1', 't2'),
    [
        (1.0, 2.0, 0.3, 1.0, None),  # r1 and t1 both given
        (None, 2.0, 0.3, None, None),  # neither r1 nor t1 given
        (1.0, 2.0, 0.3, None, 0.5),  # r2 and t2 both given
        (1.0, None, 0.3, None, None),  # neither r2 nor t2 given
    ],
)
def test_from_invalid_combination(r1, r2, f, t1, t2):
    """Test WaterPool instantiation from invalid parameter combinations."""
    with pytest.raises(ValueError):
        WaterPool(r1=r1, r2=r2, f=f, t1=t1, t2=t2)


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 't1', 't2'),
    [
        (1.0, 2.0, 3.0, None, None),  # f not between 0 and 1
        (-1.0, 2.0, 0.3, None, None),  # r1 negative
        (None, 2.0, 0.3, -1.0, None),  # t1 negative
    ],
)
def test_from_invalid_values(r1, r2, f, t1, t2):
    """Test WaterPool instantiation from invalid parameter types."""
    with pytest.raises(ValueError):
        WaterPool(r1=r1, r2=r2, f=f, t1=t1, t2=t2)


def test_slots_property():
    """Test that __slots__ is properly set."""
    a = WaterPool(r1=1.0, r2=2.0, f=0.3)
    with pytest.raises(AttributeError):
        a.new_attr = 1.0  # type: ignore


def test_dw_is_zero():
    a = WaterPool(r1=1.0, r2=2.0, f=1.0)
    assert a.dw == 0


def test_dw_cannot_be_changed():
    a = WaterPool(r1=1.0, r2=2.0, f=1)
    with pytest.raises(UserWarning):
        a.dw = 0.5
