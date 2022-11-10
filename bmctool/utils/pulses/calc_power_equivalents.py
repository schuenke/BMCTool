"""
    calc_power_equivalents.py
"""

from types import SimpleNamespace

import numpy as np


def calc_power_equivalent(rf_pulse: SimpleNamespace,
                          tp: float,
                          td: float,
                          gamma_hz: float = 42.5764) \
        -> np.ndarray:
    """
    Calculates the continuous wave power equivalent for a given rf pulse.
    :param rf_pulse: pypulseq radio-frequency pulse
    :param tp: pulse duration [s]
    :param td: interpulse delay [s]
    :param gamma_hz: gyromagnetic ratio [Hz]
    """
    amp = rf_pulse.signal / gamma_hz
    duty_cycle = tp / (tp + td)

    return np.sqrt(np.trapz(amp ** 2, rf_pulse.t) / tp * duty_cycle)  # continuous wave power equivalent


def calc_amplitude_equivalent(rf_pulse: SimpleNamespace,
                              tp: float,
                              td: float,
                              gamma_hz: float = 42.5764) \
        -> np.ndarray:
    """
    Calculates the continuous wave amplitude equivalent for a given rf pulse.
    :param rf_pulse: pypulseq radio-frequency pulse
    :param tp: pulse duration [s]
    :param td: interpulse delay [s]
    :param gamma_hz: gyromagnetic ratio [Hz]
    """
    duty_cycle = tp / (tp + td)
    alpha_rad = np.trapz(rf_pulse.signal * gamma_hz * 360, rf_pulse.t) * np.pi / 180

    return alpha_rad / (gamma_hz * 2 * np.pi * tp) * duty_cycle  # continuous wave amplitude equivalent
