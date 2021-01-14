"""
make_hypsec_half_passage.py
    Functions to create an adiabatic hyperbolic secant half passage pulse.
"""
import numpy as np
from types import SimpleNamespace
from bmctool.pypulseq.opts import Opts


def calculate_amplitude(t: np.ndarray,
                        t_0: float,
                        amp: float,
                        mu: float,
                        bandwidth: float) \
        -> np.ndarray:
    """
    Calculates amplitude modulation for a hyperbolic secant half passage pulse.
    :param t: time points of the different sample points [s]
    :param t_0: reference time point (= last point for half passage pulse) [s]
    :param amp: maximum amplitude value [µT]
    :param mu: parameter µ of hyperbolic secant pulse
    :param bandwidth: bandwidth of hyperbolic secant pulse [Hz]
    :return:
    """
    return np.divide(amp, np.cosh((bandwidth * np.pi / mu) * (t - t_0)))


def calculate_frequency(t: np.ndarray,
                        t_0: float,
                        mu: float,
                        bandwidth: float) \
        -> np.ndarray:
    """
    Calculates frequency modulation for a hyperbolic secant half passage pulse.
    :param t: time points of the different sample points [s]
    :param t_0: reference time point (= last point for half passage pulse) [s]
    :param mu: parameter µ of hyperbolic secant pulse
    :param bandwidth: bandwidth of hyperbolic secant pulse [Hz]
    :return:
    """
    beta = bandwidth * np.pi / mu
    return bandwidth * np.pi * np.tanh(beta * (t - t_0))


def calculate_phase(frequency: np.ndarray,
                    duration: float,
                    samples: int) \
        -> np.ndarray:
    """
    Calculates phase modulation of hyperbolic secant half passage pulse for given frequency modulation.
    :param frequency: frequency modulation of pulse
    :param duration: pulse duration [s]
    :param samples: number of sample points
    :return:
    """
    phase = frequency * duration / samples
    for i in range(1, samples):
        phase[i] = phase[i-1] + (frequency[i] * duration/samples)
    phase_shift = phase[-1]
    for i in range(samples):
        phase[i] = np.fmod(phase[i] - phase_shift, 2 * np.pi)
    return phase + 2 * np.pi


def make_arbitrary_rf_with_phase(signal: np.ndarray,
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

    signal = signal * flip_angle / (2 * np.pi)
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

    return rf, None


def make_hypsec_half_passage_rf(amp: float,
                                pulse_duration: float = 8e-3,
                                mu: float = 6,
                                bandwidth: float = 1200,
                                system: Opts = Opts())\
        -> SimpleNamespace:
    """
    Creates block event for an hyperbolic secant half passage pulse according to DOI: 10.1002/mrm.26370.
    :param amp: maximum amplitude value [µT]
    :param pulse_duration: pulse duration [s]
    :param mu: parameter µ of hyperbolic secant pulse
    :param bandwidth: bandwidth of hyperbolic secant pulse [Hz]
    :param system: system limits of the MR scanner
    :return:
    """

    samples = int(pulse_duration * 1e6)
    t_pulse = np.divide(np.arange(1, samples+1), samples) * pulse_duration
    t_0 = t_pulse[-1]
    w1 = calculate_amplitude(t=t_pulse, t_0=t_0, amp=1, mu=mu, bandwidth=bandwidth)
    freq = calculate_frequency(t=t_pulse, t_0=t_0, mu=mu, bandwidth=bandwidth)
    freq = freq - freq[-1]  # ensure phase ends with 0 for tip-down pulse
    phase = calculate_phase(frequency=freq, duration=pulse_duration, samples=samples)
    signal = np.multiply(w1, np.exp(1j * phase))
    flip_angle = amp * 1e-6 * system.gamma * 2 * np.pi  # factor 1e-6 converts from µT to T
    hs_half_passage, _ = make_arbitrary_rf_with_phase(signal=signal, flip_angle=flip_angle, system=system)
    return hs_half_passage
