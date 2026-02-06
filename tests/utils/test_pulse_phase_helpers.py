import numpy as np
import pypulseq as pp  # type: ignore
from bmctool.utils.pulses.calculate_phase import calculate_phase
from bmctool.utils.pulses.create_arbitrary_pulse_with_phase import create_arbitrary_pulse_with_phase


def test_calculate_phase_zero_frequency_pos_offsets_true_returns_zeros():
    samples = 8
    freq = np.zeros(samples)

    phase = calculate_phase(frequency=freq, duration=1.0, samples=samples, pos_offsets=True)

    assert phase.shape == (samples,)
    assert np.allclose(phase, 0.0)


def test_calculate_phase_zero_frequency_pos_offsets_false_returns_2pi():
    samples = 8
    freq = np.zeros(samples)

    phase = calculate_phase(frequency=freq, duration=1.0, samples=samples, pos_offsets=False)

    assert phase.shape == (samples,)
    assert np.allclose(phase, 2 * np.pi)


def test_calculate_phase_shift_idx_sets_reference_phase():
    samples = 16
    duration = 2.0
    freq = np.linspace(-10.0, 10.0, samples)

    phase_pos = calculate_phase(frequency=freq, duration=duration, samples=samples, shift_idx=-1, pos_offsets=True)
    assert 0.0 <= phase_pos[-1] < 2 * np.pi

    phase_neg = calculate_phase(frequency=freq, duration=duration, samples=samples, shift_idx=-1, pos_offsets=False)
    assert 2 * np.pi <= phase_neg[-1] < 4 * np.pi


def test_create_arbitrary_pulse_with_phase_scales_signal_and_sets_timebase():
    system = pp.Opts()
    system.rf_raster_time = 2e-6
    system.rf_dead_time = 0.0
    system.rf_ringdown_time = 0.0

    signal = np.ones(5, dtype=np.complex128)
    flip_angle = 2 * np.pi

    rf = create_arbitrary_pulse_with_phase(signal=signal, flip_angle=flip_angle, system=system)

    assert rf.type == 'rf'
    assert rf.signal.shape == (5,)
    assert rf.t.shape == (5,)
    assert np.allclose(rf.signal, 1.0)
    assert np.allclose(rf.t, np.arange(1, 6) * system.rf_raster_time)
    assert np.isclose(rf.shape_dur, 5 * system.rf_raster_time)


def test_create_arbitrary_pulse_with_phase_appends_ringdown_zeros():
    system = pp.Opts()
    system.rf_raster_time = 1e-6
    system.rf_dead_time = 0.0
    system.rf_ringdown_time = 3e-6

    signal = np.ones(4, dtype=np.complex128)
    rf = create_arbitrary_pulse_with_phase(signal=signal, flip_angle=1.0, system=system)

    assert rf.t.size == 7
    assert rf.signal.size == 7
    assert np.allclose(rf.signal[-3:], 0.0)
    assert np.all(np.diff(rf.t) > 0)
