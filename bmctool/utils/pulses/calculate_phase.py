"""
calculate_phase.py
    Function to calculate phase modulation for a given frequency modulation.
"""

import numpy as np


def calculate_phase(
    frequency: np.ndarray, duration: float, samples: int, shift_idx: int = -1, pos_offsets: bool = False
) -> np.ndarray:
    """
    calculate_phase Calculate phase modulation for a given frequency modulation.

    Returns
    -------
    np.ndarray
        Calculated phase modulation.
    """
    phase = np.zeros_like(frequency)
    for i in range(1, samples):
        phase[i] = phase[i - 1] + (frequency[i] * duration / samples)
    phase_shift = phase[shift_idx]
    for i in range(samples):
        phase[i] = np.fmod(phase[i] + 1e-12 - phase_shift, 2 * np.pi)
    if not pos_offsets:
        phase += 2 * np.pi
    return phase
