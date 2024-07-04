"""Function to create RF event with arbitrary shape and phase."""

from types import SimpleNamespace

import numpy as np
import pypulseq as pp  # type: ignore


def create_arbitrary_pulse_with_phase(
    signal: np.ndarray,
    flip_angle: float,
    freq_offset: float = 0,
    phase_offset: float = 0,
    system: pp.Opts | None = None,
) -> SimpleNamespace:
    """Create RF pulse with arbitrary shape and phase.

    Parameters
    ----------
    signal : np.ndarray
        shape of the RF pulse
    flip_angle : float
        flip angle of the RF pulse
    freq_offset : float, optional
        frequency offset of the RF pulse, by default 0
    phase_offset : float, optional
        phase offset of the RF pulse, by default 0
    system : Opts, optional
        pypulseq Opts object containing system limits, defaults to pp.Opts()
    """
    system = system or pp.Opts()

    signal *= flip_angle / (2 * np.pi)
    t = np.linspace(1, len(signal), num=len(signal)) * system.rf_raster_time

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.shape_dur = len(signal) * system.rf_raster_time
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
