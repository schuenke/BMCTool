import numpy as np
import pypulseq as pp  # type: ignore
from bmctool.utils.pulses.make_hanning import calc_hanning
from bmctool.utils.pulses.make_hanning import hanning
from bmctool.utils.pulses.make_hanning import make_gauss_hanning


def test_hanning_even_length_is_symmetric_and_in_range():
    n = 10
    w = hanning(n)

    assert w.shape == (n,)
    assert np.allclose(w, w[::-1])
    assert np.min(w) >= 0.0
    assert np.max(w) <= 1.0


def test_hanning_odd_length_is_symmetric_and_in_range():
    n = 11
    w = hanning(n)

    assert w.shape == (n,)
    assert np.allclose(w, w[::-1])
    assert np.min(w) >= 0.0
    assert np.max(w) <= 1.0


def test_calc_hanning_matches_first_half_of_hanning_even():
    n = 10
    m = n // 2

    expected = hanning(n)[:m]
    result = calc_hanning(m=m, n=n)

    assert np.allclose(result, expected)


def test_calc_hanning_matches_first_half_of_hanning_odd():
    n = 11
    m = (n + 1) // 2

    expected = hanning(n)[:m]
    result = calc_hanning(m=m, n=n)

    assert np.allclose(result, expected)


def test_make_gauss_hanning_normalizes_area_to_flip_angle():
    flip_angle = 1.234
    pulse_duration = 2e-3
    system = pp.Opts()

    rf = make_gauss_hanning(flip_angle=flip_angle, pulse_duration=pulse_duration, system=system)

    assert rf.signal.size == rf.t.size

    area = np.trapezoid(rf.signal, x=rf.t)
    expected_area = flip_angle / (2 * np.pi)
    assert np.isclose(area, expected_area, rtol=1e-3, atol=1e-6)
