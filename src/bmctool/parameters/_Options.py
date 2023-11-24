"""Definition of Options dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class Options:
    """Class to store additional options.

    Parameters:
    -----------
    verbose : bool, optional
        Flag to activate detailed outputs, by default False
    reset_init_mag : bool, optional
        flag to reset the initial magnetization for every offset, by default True
    scale : float, optional
        value of initial magnetization if reset_init_mag is True, by default 1.0
    max_pulse_samples : int, optional
        maximum number of simulation steps for one RF pulse, by default 500
    """

    verbose: bool = False
    reset_init_mag: bool = True
    scale: float = 1.0
    max_pulse_samples: int = 500
