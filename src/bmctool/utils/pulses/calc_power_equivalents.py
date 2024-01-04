"""calc_power_equivalents.py."""

from types import SimpleNamespace

import numpy as np

from bmctool import GAMMA_HZ


def calc_power_equivalent(rf_pulse: SimpleNamespace, tp: float, td: float, gamma_hz: float = GAMMA_HZ) -> float:
    """calc_power_equivalent Calculate continuous wave power equivalent for a
    given rf pulse.

    Parameters
    ----------
    rf_pulse : SimpleNamespace
        PyPulseq rf pulse object.
    tp : float
        RF pulse duration.
    td : float
        interpulse delay.
    gamma_hz : float, optional
        gyromagnetic ratio, by default GAMMA_HZ

    Returns
    -------
    float
        Continuous wave power equivalent value.
    """
    amp = rf_pulse.signal / gamma_hz
    duty_cycle = tp / (tp + td)

    return np.sqrt(np.trapz(amp**2, rf_pulse.t) / tp * duty_cycle)


def calc_amplitude_equivalent(rf_pulse: SimpleNamespace, tp: float, td: float, gamma_hz: float = GAMMA_HZ) -> float:
    """calc_amplitude_equivalent Calculate continuous wave amplitude equivalent
    for a given rf pulse.

    Parameters
    ----------
    rf_pulse : SimpleNamespace
        PyPulseq rf pulse object.
    tp : float
        RF pulse duration.
    td : float
        interpulse delay.
    gamma_hz : float, optional
        gyromagnetic ratio, by default GAMMA_HZ

    Returns
    -------
    float
        Continuous wave amplitude equivalent value.
    """
    duty_cycle = tp / (tp + td)
    alpha_rad = np.trapz(rf_pulse.signal * gamma_hz * 360, rf_pulse.t) * np.pi / 180

    return alpha_rad / (gamma_hz * 2 * np.pi * tp) * duty_cycle
