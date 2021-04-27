"""
create_arbitrary_pulse_with_phase.py
    Function to create a radio-frequency pulse event with arbitrary pulse shape and phase modulation.
"""

import numpy as np
from types import SimpleNamespace
from pypulseq.opts import Opts


def create_arbitrary_pulse_with_phase(signal: np.ndarray,
                                      flip_angle: float,
                                      freq_offset: float = 0,
                                      phase_offset: float = 0,
                                      system: Opts = Opts()) \
        -> (SimpleNamespace, None):
    """
    Creates a radio-frequency pulse event with arbitrary pulse shape and phase modulation
    :param signal: signal modulation (amplitude and phase) of pulse
    :param flip_angle: flip angle of pulse [rad]
    :param freq_offset: frequency offset [Hz]
    :param phase_offset: phase offset [rad]
    :param system: system limits of the MR scanner
    :return:
    """

    signal *= (flip_angle / (2 * np.pi))
    t = np.linspace(1, len(signal)) * system.rf_raster_time

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = system.rf_dead_time

    if rf.ringdown_time > 0:
        t_fill = np.arange(1, round(rf.ringdown_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, rf.t[-1] + t_fill))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    return rf
