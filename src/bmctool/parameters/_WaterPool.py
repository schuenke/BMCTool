"""Definition of WaterPool dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class WaterPool:
    """Class to store water pool parameters.

    Parameters:
    -----------
    r1 : float
        R1 relaxation rate [Hz] (1/T1)
    r2 : float
        R2 relaxation rate [Hz] (1/T2)
    f : float, optional
        Pool size fraction, by default 1
    """

    r1: float
    r2: float
    f: float = 1
