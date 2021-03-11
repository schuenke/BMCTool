import numpy as np
from types import SimpleNamespace
from scipy.signal import hanning
from pypulseq.opts import Opts
from pypulseq.make_gauss_pulse import make_gauss_pulse


def make_gauss_hanning(flip_angle: float,
                       pulse_duration: float,
                       system: Opts = Opts())\
        -> SimpleNamespace:
    """
    Creates a radio-frequency pulse event for a gauss pulse with hanning window.
    :param flip_angle: flip_angle of the rf pulse
    :param pulse_duration: rf pulse duration [s]
    :param system: system limits of the MR scanner
    """

    rf_pulse, _, _ = make_gauss_pulse(flip_angle=flip_angle, duration=pulse_duration, system=system)
    n_signal = np.sum(np.abs(rf_pulse.signal) > 0)
    hanning_shape = hanning(n_signal + 2)
    rf_pulse.signal[:n_signal] = hanning_shape[1:-1] / np.trapz(rf_pulse.t[:n_signal], hanning_shape[1:-1]) * \
                                 (flip_angle / (2 * np.pi))
    return rf_pulse
