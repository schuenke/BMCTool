import pytest
from bmctool.utils.misc import truthy_check


@pytest.mark.parametrize(
    ('parameter', 'value'),
    [
        (True, True),
        (1, True),
        (1.0, True),
        ('True', True),
        ('true', True),
        (False, False),
        (0, False),
        ('False', False),
        ('false', False),
    ],
)
def test_truthy_check_true(parameter, value):
    """Test that truthy_check() returns correct boolean."""
    assert truthy_check(parameter) == value
