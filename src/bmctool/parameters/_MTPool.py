"""Definition of MTPool dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class MTPool:
    """Class to store MT pool parameters.

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
    lineshape : str
        Lineshape of the MT pool ("Lorentzian", "SuperLorentzian" or "None")
    """

    r1: float
    r2: float
    k: float
    f: float
    dw: float
    lineshape: str
