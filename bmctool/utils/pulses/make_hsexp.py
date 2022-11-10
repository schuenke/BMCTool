"""
make_hsexp.py
    Function to create all possible HSExp pulses (tip-down/tip-up & pos/neg offset)
"""

from types import SimpleNamespace

import numpy as np
from pypulseq import Opts

from bmctool.utils.pulses.calculate_phase import calculate_phase
from bmctool.utils.pulses.create_arbitrary_pulse_with_phase import create_arbitrary_pulse_with_phase
from bmctool.utils.pulses.make_hypsec_half_passage import calculate_amplitude as hypsec_amp


def calculate_window_modulation(t: np.ndarray,
                                t0: float) \
        -> np.ndarray:
    """
    Calculates modulation function for HSExp pulses.
    :param t: time points of the different sample points [s]
    :param t0: reference time point (= last point for half passage pulse) [s]
    :return:
    """
    return 0.42 - 0.5 * np.cos(np.pi * t / t0) + 0.08 * np.cos(2 * np.pi * t / t0)


def calculate_frequency(t: np.ndarray,
                        t0: float,
                        bandwidth: float,
                        ef: float,
                        freq_factor: int) \
        -> np.ndarray:
    """
    Calculates modulation function for HSExp pulses.
    :param t: time points of the different sample points [s]
    :param t0: reference time point (= last point for half passage pulse) [s]
    :param bandwidth: bandwidth of the pulse [Hz]
    :param ef: dimensionless parameter to control steepness of the exponential curve
    :param freq_factor: factor (-1 or +1) to switch between positive and negative offsets
    """

    return -freq_factor * bandwidth * np.pi * np.exp(-t / t0 * ef)


def make_hsexp(amp: float = 1.0,
               t_p: float = 12e-3,
               mu: float = 65,
               bandwidth: float = 2500,
               t_window: float = 3.5e-3,
               ef: float = 3.5,
               tip_down: bool = True,
               pos_offset: bool = True,
               system: Opts = Opts(),
               gamma_hz: float = 42.5764) \
        -> SimpleNamespace:
    """
    Creates a radio-frequency pulse event with amplitude and phase modulation of a HSExp pulse.
    :param amp: maximum amplitude value [µT]
    :param t_p: pulse pulse_duration [s]
    :param mu: parameter µ of hyperbolic secant pulse
    :param bandwidth: bandwidth of hyperbolic secant pulse [Hz]
    :param t_window: pulse_duration of window function
    :param ef: dimensionless parameter to control steepness of the exponential curve
    :param tip_down: flag to switch between tip down (True) and tip up (False) pulses
    :param pos_offset: flag to switch between positive (True) and negative (False) offsets
    :param system: system limits of the MR scanner
    :param gamma_hz: gyromagnetic ratio [Hz]
    """

    samples = int(t_p * 1e6)
    t_pulse = np.divide(np.arange(1, samples + 1), samples) * t_p  # time point array

    # find start index of window function
    idx_window = np.argmin(np.abs(t_pulse - t_window))

    if tip_down:
        shift_idx = -1
    else:
        shift_idx = 0

    # calculate amplitude of hyperbolic secant (HS) pulse
    w1 = hypsec_amp(t_pulse, t_pulse[shift_idx], amp, mu, bandwidth)

    # calculate and apply modulation function to convert HS into HSExp pulse
    window_mod = calculate_window_modulation(t_pulse[:idx_window], t_pulse[idx_window])
    if tip_down:
        w1[:idx_window] = w1[:idx_window] * window_mod
    else:
        w1[-idx_window:] = w1[-idx_window:] * np.flip(window_mod)

    # calculate freq modulation of pulse
    if tip_down and pos_offset:
        dfreq = calculate_frequency(t_pulse, t_pulse[-1], bandwidth, ef, 1)
    elif tip_down and not pos_offset:
        dfreq = calculate_frequency(t_pulse, t_pulse[-1], bandwidth, ef, -1)
    elif not tip_down and pos_offset:
        dfreq = calculate_frequency(np.flip(t_pulse), t_pulse[-1], bandwidth, ef, 1)
    elif not tip_down and not pos_offset:
        dfreq = calculate_frequency(np.flip(t_pulse), t_pulse[-1], bandwidth, ef, -1)

    # make freq modulation end (in case of tip-down) or start (in case of tip-up) with dw = 0
    diff_idx = np.argmin(np.abs(dfreq))
    dfreq -= dfreq[diff_idx]

    # calculate phase (= integrate over dfreq)
    dphase = calculate_phase(dfreq, t_p, samples, shift_idx=shift_idx, pos_offsets=pos_offset)

    # create pypulseq rf pulse object
    signal = w1 * np.exp(1j * dphase)  # create complex array with amp and phase
    flip_angle = gamma_hz * 2 * np.pi
    hsexp = create_arbitrary_pulse_with_phase(signal=signal, flip_angle=flip_angle, system=system)

    return hsexp


def generate_hsexp_dict(amp: float = 1.0,
                        t_p: float = 12e-3,
                        mu: float = 65,
                        bandwidth: float = 2500,
                        t_window: float = 3.5e-3,
                        ef: float = 3.5,
                        system: Opts = Opts(),
                        gamma_hz: float = 42.5764) \
        -> dict:
    """
    Creates a dictionary with the 4 different hsexp pulses (tip-down/up and pos/neg offsets)
    :param amp: maximum amplitude value [µT]
    :param t_p: pulse pulse_duration [s]
    :param mu: parameter µ of hyperbolic secant pulse
    :param bandwidth: bandwidth of hyperbolic secant pulse [Hz]
    :param t_window: pulse_duration of window function
    :param ef: dimensionless parameter to control steepness of the exponential curve
    :param system: system limits of the MR scanner
    :param gamma_hz: gyromagnetic ratio [Hz]
    :return:
    """

    pulse_dict = {}  # create empty dict for the 4 different pulses

    # tip-down positive offset
    pre_pos = make_hsexp(amp=amp,
                         t_p=t_p,
                         mu=mu,
                         bandwidth=bandwidth,
                         t_window=t_window,
                         ef=ef,
                         tip_down=True,
                         pos_offset=True,
                         system=system,
                         gamma_hz=gamma_hz)

    pulse_dict.update({'pre_pos': pre_pos})

    # tip-down negative offset
    pre_neg = make_hsexp(amp=amp,
                         t_p=t_p,
                         mu=mu,
                         bandwidth=bandwidth,
                         t_window=t_window,
                         ef=ef,
                         tip_down=True,
                         pos_offset=False,
                         system=system,
                         gamma_hz=gamma_hz)

    pulse_dict.update({'pre_neg': pre_neg})

    # tip-up positive offsets
    post_pos = make_hsexp(amp=amp,
                          t_p=t_p,
                          mu=mu,
                          bandwidth=bandwidth,
                          t_window=t_window,
                          ef=ef,
                          tip_down=False,
                          pos_offset=True,
                          system=system,
                          gamma_hz=gamma_hz)

    pulse_dict.update({'post_pos': post_pos})

    # tip-up negative offsets
    post_neg = make_hsexp(amp=amp,
                          t_p=t_p,
                          mu=mu,
                          bandwidth=bandwidth,
                          t_window=t_window,
                          ef=ef,
                          tip_down=False,
                          pos_offset=False,
                          system=system,
                          gamma_hz=gamma_hz)

    pulse_dict.update({'post_neg': post_neg})

    return pulse_dict
