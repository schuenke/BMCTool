"""
make_hypsec_half_passage.py
    Functions to create an adiabatic hyperbolic secant half passage pulse.
"""
from types import SimpleNamespace

import numpy as np
from pypulseq import Opts

from bmctool.utils.pulses.calculate_phase import calculate_phase
from bmctool.utils.pulses.create_arbitrary_pulse_with_phase import create_arbitrary_pulse_with_phase


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
    """
    beta = bandwidth * np.pi / mu
    return bandwidth * np.pi * np.tanh(beta * (t - t_0))


def make_hypsec_half_passage_rf(amp: float,
                                pulse_duration: float = 8e-3,
                                mu: float = 6,
                                bandwidth: float = 1200,
                                system: Opts = Opts()) \
        -> SimpleNamespace:
    """
    Creates block event for an hyperbolic secant half passage pulse according to DOI: 10.1002/mrm.26370.
    :param amp: maximum amplitude value [µT]
    :param pulse_duration: pulse pulse_duration [s]
    :param mu: parameter µ of hyperbolic secant pulse
    :param bandwidth: bandwidth of hyperbolic secant pulse [Hz]
    :param system: system limits of the MR scanner
    """

    samples = int(pulse_duration * 1e6)
    t_pulse = np.divide(np.arange(1, samples + 1), samples) * pulse_duration
    t_0 = t_pulse[-1]
    w1 = calculate_amplitude(t=t_pulse, t_0=t_0, amp=1, mu=mu, bandwidth=bandwidth)
    freq = calculate_frequency(t=t_pulse, t_0=t_0, mu=mu, bandwidth=bandwidth)
    freq = freq - freq[-1]  # ensure phase ends with 0 for tip-down pulse
    phase = calculate_phase(frequency=freq, duration=pulse_duration, samples=samples)
    signal = np.multiply(w1, np.exp(1j * phase))
    flip_angle = amp * 1e-6 * system.gamma * 2 * np.pi  # factor 1e-6 converts from µT to T
    hs_half_passage = create_arbitrary_pulse_with_phase(signal=signal, flip_angle=flip_angle, system=system)
    return hs_half_passage
