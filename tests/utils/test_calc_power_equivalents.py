from types import SimpleNamespace

import numpy as np
from bmctool.utils.pulses.calc_power_equivalents import calc_amplitude_equivalent
from bmctool.utils.pulses.calc_power_equivalents import calc_power_equivalent


def test_calc_power_equivalent_constant_signal():
    t = np.linspace(0.0, 2.0, 1001)
    tp = t[-1] - t[0]
    td = 1.0
    gamma_hz = 2.0
    signal = np.full_like(t, 4.0)
    rf = SimpleNamespace(signal=signal, t=t)

    duty_cycle = tp / (tp + td)
    expected = abs(4.0) / gamma_hz * np.sqrt(duty_cycle)

    result = calc_power_equivalent(rf_pulse=rf, tp=tp, td=td, gamma_hz=gamma_hz)
    assert np.isclose(result, expected)


def test_calc_power_equivalent_nonnegative():
    t = np.linspace(0.0, 1.0, 501)
    tp = 1.0
    td = 0.0
    gamma_hz = 3.0
    signal = -np.ones_like(t) * 6.0
    rf = SimpleNamespace(signal=signal, t=t)

    result = calc_power_equivalent(rf_pulse=rf, tp=tp, td=td, gamma_hz=gamma_hz)
    assert result >= 0.0


def test_calc_amplitude_equivalent_constant_signal():
    t = np.linspace(0.0, 2.0, 1001)
    tp = t[-1] - t[0]
    td = 1.0
    gamma_hz = 2.0
    signal = np.full_like(t, 4.0)
    rf = SimpleNamespace(signal=signal, t=t)

    duty_cycle = tp / (tp + td)
    expected = 4.0 * duty_cycle

    result = calc_amplitude_equivalent(rf_pulse=rf, tp=tp, td=td, gamma_hz=gamma_hz)
    assert np.isclose(result, expected)


def test_calc_amplitude_equivalent_preserves_sign():
    t = np.linspace(0.0, 1.0, 501)
    tp = 1.0
    td = 0.5
    gamma_hz = 10.0
    signal = np.full_like(t, -2.0)
    rf = SimpleNamespace(signal=signal, t=t)

    result = calc_amplitude_equivalent(rf_pulse=rf, tp=tp, td=td, gamma_hz=gamma_hz)
    assert result < 0.0


def test_calc_amplitude_equivalent_matches_trapezoid_definition():
    t = np.linspace(0.0, 1.0, 2001)
    tp = 1.0
    td = 0.25
    gamma_hz = 7.0
    signal = np.sin(2 * np.pi * t) + 0.5
    rf = SimpleNamespace(signal=signal, t=t)

    duty_cycle = tp / (tp + td)
    alpha_rad = np.trapezoid(signal * gamma_hz * 360, t) * np.pi / 180
    expected = alpha_rad / (gamma_hz * 2 * np.pi * tp) * duty_cycle

    result = calc_amplitude_equivalent(rf_pulse=rf, tp=tp, td=td, gamma_hz=gamma_hz)
    assert np.isclose(result, expected)
