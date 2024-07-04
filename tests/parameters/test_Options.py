import pytest
from bmctool.parameters.Options import Options


@pytest.mark.parametrize(
    ('verbose', 'reset_init_mag', 'scale', 'max_pulse_samples'),
    [
        (True, False, 0.5, 100),  # all values correct type
        ('True', False, 0.5, 100),  # verbose str instead of bool
        (1, False, 0.5, 100),  # verbose int instead of bool
        (True, False, 0.5, 100.0),  # max_pulse_samples float instead of int
        (True, False, 0.5, '100'),  # max_pulse_samples str instead of int
        (True, True, '5e-1', 100),  # scale str (scientific notation)
    ],
)
def test_from_valid_params(verbose, reset_init_mag, scale, max_pulse_samples):
    """Test Options instantiation from valid parameters."""
    a = Options(
        verbose=verbose,
        reset_init_mag=reset_init_mag,
        scale=scale,
        max_pulse_samples=max_pulse_samples,
    )

    # assert that the attributes are set correctly
    assert a.verbose == bool(verbose)
    assert a.reset_init_mag == bool(reset_init_mag)
    assert a.scale == float(scale)
    assert a.max_pulse_samples == int(max_pulse_samples)


@pytest.mark.parametrize(
    ('verbose', 'reset_init_mag', 'scale', 'max_pulse_samples'),
    [
        (True, False, 1.5, 100),  # scale not between 0 and 1
        ('True', False, 0.5, -10),  # max_pulse_samples negative
        ('5', False, 0.5, 100),  # verbose str (not convertible to bool)
    ],
)
def test_raise_valueerror_for_invalid_values(verbose, reset_init_mag, scale, max_pulse_samples):
    """Test Options instantiation from invalid parameter types."""
    with pytest.raises(ValueError):
        Options(
            verbose=verbose,
            reset_init_mag=reset_init_mag,
            scale=scale,
            max_pulse_samples=max_pulse_samples,
        )
