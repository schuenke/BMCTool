import numpy as np
import pypulseq as pp  # type: ignore
from bmctool.utils.pulses.make_hypsec_half_passage import calculate_amplitude
from bmctool.utils.pulses.make_hypsec_half_passage import calculate_frequency
from bmctool.utils.pulses.make_hypsec_half_passage import make_hypsec_half_passage_rf


def test_calculate_amplitude_is_max_at_t0_and_symmetric_around_t0():
    t0 = 1.0
    t = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    amp = 2.5
    mu = 6.0
    bandwidth = 1200.0

    w1 = calculate_amplitude(t=t, t_0=t0, amp=amp, mu=mu, bandwidth=bandwidth)

    assert np.isclose(w1[2], amp)
    assert np.all(w1 <= amp)
    assert np.isclose(w1[0], w1[-1])
    assert np.isclose(w1[1], w1[-2])


def test_calculate_frequency_crosses_zero_at_t0_and_is_antisymmetric():
    t0 = 1.0
    t = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    mu = 6.0
    bandwidth = 1200.0

    df = calculate_frequency(t=t, t_0=t0, mu=mu, bandwidth=bandwidth)

    assert np.isclose(df[2], 0.0)
    assert np.isclose(df[0], -df[-1])
    assert np.isclose(df[1], -df[-2])


def test_make_hypsec_half_passage_rf_returns_complex_signal_and_monotonic_time():
    system = pp.Opts()

    rf = make_hypsec_half_passage_rf(amp=1.0, pulse_duration=200e-6, mu=6.0, bandwidth=1200.0, system=system)

    assert rf.type == 'rf'
    assert rf.signal.size == rf.t.size
    assert np.iscomplexobj(rf.signal)
    assert np.all(np.diff(rf.t) > 0)


def test_make_hypsec_half_passage_rf_signal_magnitude_decays_from_end():
    system = pp.Opts()

    rf = make_hypsec_half_passage_rf(amp=1.0, pulse_duration=400e-6, mu=6.0, bandwidth=1200.0, system=system)

    mag = np.abs(rf.signal)
    assert mag[0] < np.max(mag)
