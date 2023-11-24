"""Definition of System dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class System:
    """Class to store system parameters.

    Parameters:
    -----------
    b0 : float
        field strength [T]
    gamma : float
        gyromagnetic ratio [MHz/T]
    b0_inhom : float
        _description_
    rel_b1 : float
        _description_
    """

    b0: float
    gamma: float
    b0_inhom: float
    rel_b1: float
