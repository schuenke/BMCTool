from types import SimpleNamespace

import numpy as np
from pypulseq import Opts
from pypulseq import make_gauss_pulse


def hanning(n: int) -> np.ndarray:
    """
    Returns a symmetric n point hanning window.
    :param n: number of points
    :return: n point window
    """
    if n % 2 == 0:
        half = n / 2
        window = calc_hanning(half, n)
        window = np.append(window, np.flipud(window))
    else:
        half = (n + 1) / 2
        window = calc_hanning(half, n)
        window = np.append(window, np.flipud(window[:-1]))
    return window


def calc_hanning(m: int,
                 n: int) \
        -> np.ndarray:
    """
    Calculates and returns the first m points of an n point hanning window.
    """
    window = .5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (n + 1)))
    return window


def make_gauss_hanning(flip_angle: float,
                       pulse_duration: float,
                       system: Opts = Opts()) \
        -> SimpleNamespace:
    """
    Creates a radio-frequency pulse event for a gauss pulse with hanning window.
    :param flip_angle: flip_angle of the rf pulse
    :param pulse_duration: rf pulse duration [s]
    :param system: system limits of the MR scanner
    """

    rf_pulse = make_gauss_pulse(flip_angle=flip_angle, duration=pulse_duration, system=system, phase_offset=0)
    # n_signal = np.sum(np.abs(rf_pulse.signal) > 0)
    n_signal = rf_pulse.signal.size
    # hanning_shape = hanning(n_signal + 2)
    hanning_shape = hanning(n_signal)
    # rf_pulse.signal[:n_signal] = hanning_shape[1:-1] / np.trapz(rf_pulse.t[:n_signal], hanning_shape[1:-1]) * \
    #                              (flip_angle / (2 * np.pi))
    rf_pulse.signal = hanning_shape / np.trapz(hanning_shape, x=rf_pulse.t) * (flip_angle / (2 * np.pi))
    return rf_pulse
