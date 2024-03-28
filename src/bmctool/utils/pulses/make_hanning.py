"""Functions to create a RF pulse event for a gaussian pulse with hanning window."""

from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from pypulseq import make_gauss_pulse


def hanning(n: int) -> np.ndarray:
    """Return a symmetric n point hanning window."""
    if n % 2 == 0:
        half = int(n / 2)
        window = calc_hanning(half, n)
        window = np.append(window, np.flipud(window))
    else:
        half = int((n + 1) / 2)
        window = calc_hanning(half, n)
        window = np.append(window, np.flipud(window[:-1]))
    return window


def calc_hanning(m: int, n: int) -> np.ndarray:
    """Calculate the first m points of an n point hanning window."""
    window = np.array(0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (n + 1))))
    return window


def make_gauss_hanning(
    flip_angle: float,
    pulse_duration: float,
    system: pp.Opts | None = None,
) -> SimpleNamespace:
    """Create an RF pulse event for a gaussian pulse with hanning window.

    Parameter
    ---------
    flip_angle
        flip angle of the pulse [rad]
    pulse_duration
        duration of the pulse [s]
    system
        system limits of the MR scanner, by default pp.Opts()

    Return
    ------
    SimpleNamespace
        RF pulse event for a gaussian pulse with hanning window.
    """
    system = system or pp.Opts()

    rf_pulse = make_gauss_pulse(flip_angle=flip_angle, duration=pulse_duration, system=system, phase_offset=0)
    n_signal = rf_pulse.signal.size
    hanning_shape = hanning(n_signal)
    rf_pulse.signal = hanning_shape / np.trapz(hanning_shape, x=rf_pulse.t) * (flip_angle / (2 * np.pi))
    return rf_pulse  # type: ignore
