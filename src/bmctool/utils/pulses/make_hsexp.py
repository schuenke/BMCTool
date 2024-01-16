from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from bmctool import GAMMA_HZ
from bmctool.utils.pulses.calculate_phase import calculate_phase
from bmctool.utils.pulses.create_arbitrary_pulse_with_phase import (
    create_arbitrary_pulse_with_phase,
)
from bmctool.utils.pulses.make_hypsec_half_passage import (
    calculate_amplitude as hypsec_amp,
)


def calculate_window_modulation(t: np.ndarray, t0: float) -> np.ndarray:
    """Calculate modulation function for HSExp pulses.

    Parameter
    ---------
    t
        time points of the different sample points [s]
    t0
        reference time point (= last point for half passage pulse) [s]

    Return
    ------
    np.ndarray
        Calculated window function.
    """

    return 0.42 - 0.5 * np.cos(np.pi * t / t0) + 0.08 * np.cos(2 * np.pi * t / t0)


def calculate_frequency(
    t: np.ndarray,
    t0: float,
    bandwidth: float,
    ef: float,
    freq_factor: int,
) -> np.ndarray:
    """Calculate modulation function for HSExp pulses.

    Parameter
    ---------
    t
        time points of the different sample points [s]
    t0
        reference time point (= last point for half passage pulse) [s]
    bandwidth
        bandwidth of the pulse [Hz]
    ef
        dimensionless parameter to control steepness of the exponential curve
    freq_factor
        factor (-1 or +1) to switch between positive and negative offsets

    Return
    ------
    np.ndarray
        Calculated modulation function for HSExp pulse.
    """

    return -freq_factor * bandwidth * np.pi * np.exp(-t / t0 * ef)


def make_hsexp(
    amp: float = 1.0,
    t_p: float = 12e-3,
    mu: float = 65,
    bandwidth: float = 2500,
    t_window: float = 3.5e-3,
    ef: float = 3.5,
    tip_down: bool = True,
    pos_offset: bool = True,
    system: pp.Opts | None = None,
    gamma_hz: float = GAMMA_HZ,
) -> SimpleNamespace:
    """Create HSExp RF pulse using given settings.

    Parameter
    ---------
    amp, optional
        maximum amplitude value [µT], by default 1.0
    t_p, optional
        pulse pulse_duration [s], by default 12e-3
    mu, optional
        parameter µ of hyperbolic secant pulse, by default 65
    bandwidth, optional
        bandwidth of hyperbolic secant pulse [Hz], by default 2500
    t_window, optional
        duration of window function, by default 3.5e-3
    ef, optional
        dimensionless parameter to control steepness of the exponential curve, by default 3.5
    tip_down, optional
        flag to switch between tip down (True) and tip up (False) pulses, by default True
    pos_offset, optional
        flag to switch between positive (True) and negative (False) offsets, by default True
    system, optional
        system limits of the MR scanner, defaults to pp.Opts()
    gamma_hz, optional
        gyromagnetic ratio [Hz], by default GAMMA_HZ

    Return
    ------
    SimpleNamespace
        PyPulseq rf pulse object.
    """

    system = system or pp.Opts()

    samples = int(t_p * 1e6)
    t_pulse = np.divide(np.arange(1, samples + 1), samples) * t_p  # time point array

    # find start index of window function
    idx_window = np.argmin(np.abs(t_pulse - t_window))

    shift_idx = -1 if tip_down else 0

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


def generate_hsexp_dict(
    amp: float = 1.0,
    t_p: float = 12e-3,
    mu: float = 65,
    bandwidth: float = 2500,
    t_window: float = 3.5e-3,
    ef: float = 3.5,
    system: pp.Opts | None = None,
    gamma_hz: float = GAMMA_HZ,
) -> dict:
    """Create a dict with all 4 possible HSexp RF pulses.

    Possible combinations are:
    - tip-down positive offset
    - tip-down negative offset
    - tip-up positive offset
    - tip-up negative offset

    Parameter
    ---------
    amp, optional
        maximum amplitude value [µT], by default 1.0
    t_p, optional
        pulse pulse_duration [s], by default 12e-3
    mu, optional
        parameter µ of hyperbolic secant pulse, by default 65
    bandwidth, optional
        bandwidth of hyperbolic secant pulse [Hz], by default 2500
    t_window, optional
        duration of window function, by default 3.5e-3
    ef, optional
        dimensionless parameter to control steepness of the exponential curve, by default 3.5
    system, optional
        system limits of the MR scanner, by default pp.Opts()
    gamma_hz, optional
        gyromagnetic ratio [Hz], by default GAMMA_HZ

    Return
    ------
    dict
        dict with all 4 possible HSexp RF pulses
    """

    system = system or pp.Opts()

    pulse_dict = {}  # create empty dict for the 4 different pulses

    # tip-down positive offset
    pre_pos = make_hsexp(
        amp=amp,
        t_p=t_p,
        mu=mu,
        bandwidth=bandwidth,
        t_window=t_window,
        ef=ef,
        tip_down=True,
        pos_offset=True,
        system=system,
        gamma_hz=gamma_hz,
    )

    pulse_dict.update({'pre_pos': pre_pos})

    # tip-down negative offset
    pre_neg = make_hsexp(
        amp=amp,
        t_p=t_p,
        mu=mu,
        bandwidth=bandwidth,
        t_window=t_window,
        ef=ef,
        tip_down=True,
        pos_offset=False,
        system=system,
        gamma_hz=gamma_hz,
    )

    pulse_dict.update({'pre_neg': pre_neg})

    # tip-up positive offsets
    post_pos = make_hsexp(
        amp=amp,
        t_p=t_p,
        mu=mu,
        bandwidth=bandwidth,
        t_window=t_window,
        ef=ef,
        tip_down=False,
        pos_offset=True,
        system=system,
        gamma_hz=gamma_hz,
    )

    pulse_dict.update({'post_pos': post_pos})

    # tip-up negative offsets
    post_neg = make_hsexp(
        amp=amp,
        t_p=t_p,
        mu=mu,
        bandwidth=bandwidth,
        t_window=t_window,
        ef=ef,
        tip_down=False,
        pos_offset=False,
        system=system,
        gamma_hz=gamma_hz,
    )

    pulse_dict.update({'post_neg': post_neg})

    return pulse_dict
