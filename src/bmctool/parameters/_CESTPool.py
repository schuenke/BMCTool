"""Definition of CESTPool dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class CESTPool:
    """Class to store CEST pool parameters.

    Parameters:
    -----------
    r1 : float
        R1 relaxation rate [Hz] (1/T1)
    r2 : float
        R2 relaxation rate [Hz] (1/T2)
    k : float
        exchange rate [Hz]
    f : float
        pool size fraction
    dw : float
        chemical shift from water [ppm]
    """

    r1: float
    r2: float
    k: float
    f: float
    dw: float
