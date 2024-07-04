import pytest
from bmctool.parameters.Pool import Pool


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 'dw', 't1', 't2'),
    [
        (1.0, 2.0, 0.3, 5.0, None, None),  # all values float
        (1.0, 2.0, 0.3, 5, None, None),  # dw int
        (1.0, 2.0, 0.3, '5', None, None),  # dw str
        (None, 2.0, 0.3, 5.0, 1.0, None),  # r1 None, but t1 float
        (None, 2.0, 0.3, 5.0, '1.0', None),  # r1 None, but t1 str
        (1.0, None, 0.3, 5.0, None, 2.0),  # r2 None, but t2 float
        (1.0, 2, '1e-1', 5.0, None, None),  # f str (scientific notation)
    ],
)
def test_from_valid_params(r1, r2, f, dw, t1, t2):
    """Test Pool instantiation from valid parameters."""
    a = Pool(r1=r1, r2=r2, f=f, dw=dw, t1=t1, t2=t2)

    # assert that the attributes are set correctly
    assert a.r1 == float(r1) if r1 is not None else 1 / float(t1)
    assert a.r2 == float(r2) if r2 is not None else 1 / float(t2)
    assert a.f == float(f)
    assert a.dw == float(dw)


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 'dw', 't1', 't2'),
    [
        (1.0, 2.0, 0.3, 4.0, 1.0, None),  # r1 and t1 both given
        (None, 2.0, 0.3, 4.0, None, None),  # neither r1 nor t1 given
        (1.0, 2.0, 0.3, 4.0, None, 0.5),  # r2 and t2 both given
        (1.0, None, 0.3, 4.0, None, None),  # neither r2 nor t2 given
    ],
)
def test_from_invalid_combination(r1, r2, f, dw, t1, t2):
    """Test Pool instantiation from invalid parameter combinations."""
    with pytest.raises(ValueError):
        Pool(r1=r1, r2=r2, f=f, dw=dw, t1=t1, t2=t2)


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 'dw', 't1', 't2'),
    [
        ([1.0], 2.0, 0.3, 4.0, None, None),  # invalid type list
        ((1.0,), 2.0, 0.3, 4.0, None, None),  # invalid type tuple))
    ],
)
def test_from_invalid_types(r1, r2, f, dw, t1, t2):
    """Test Pool instantiation from invalid parameter types."""
    with pytest.raises(TypeError):
        Pool(r1=r1, r2=r2, f=f, dw=dw, t1=t1, t2=t2)


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 'dw', 't1', 't2'),
    [
        (1.0, 2.0, 3.0, 4.0, None, None),  # f not between 0 and 1
        (-1.0, 2.0, 0.3, 4.0, None, None),  # r1 negative
        (None, 2.0, 0.3, 4.0, -1.0, None),  # t1 negative
    ],
)
def test_from_invalid_values(r1, r2, f, dw, t1, t2):
    """Test Pool instantiation from invalid parameter types."""
    with pytest.raises(ValueError):
        Pool(r1=r1, r2=r2, f=f, dw=dw, t1=t1, t2=t2)


@pytest.mark.parametrize(
    ('r1', 'r2', 'f', 'dw', 't1', 't2'),
    [
        ('1,0', 2.0, 0.3, 5.0, None, None),  # r1 str not convertible to float
        (None, 2.0, 0.3, 5.0, '1,0', None),  # t1 str not convertible to float
    ],
)
def test_from_unconvertable_strings(r1, r2, f, dw, t1, t2):
    """Test Pool instantiation from invalid parameter types."""
    with pytest.raises(ValueError):
        Pool(r1=r1, r2=r2, f=f, dw=dw, t1=t1, t2=t2)


def test_slots_property():
    """Test that __slots__ is properly set."""
    a = Pool(r1=1.0, r2=2.0, f=0.3, dw=4.0)
    with pytest.raises(AttributeError):
        a.new_attr = 1.0


def test_from_dict_classmethod():
    """Test CESTPool instantiation from a dictionary."""
    d = {'r1': 1.0, 'r2': 2.0, 'f': 0.3, 'dw': 4.0}
    a = Pool.from_dict(d)
    b = Pool(**d)
    assert a == b
