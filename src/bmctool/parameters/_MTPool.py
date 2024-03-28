"""Definition of MTPool class."""

from __future__ import annotations

import typing

from bmctool.parameters import Pool


class MTPool(Pool):
    """Class to store MTPool parameters."""

    valid_lineshapes: typing.ClassVar[list[str]] = ['lorentzian', 'superlorentzian']
    __slots__ = ['_r1', '_r2', '_k', '_f', '_dw', '_lineshape']

    def __init__(
        self,
        k: float,
        f: float,
        dw: float,
        lineshape: str,
        r1: float | None = None,
        r2: float | None = None,
        t1: float | None = None,
        t2: float | None = None,
    ):
        """Initialize MTPool object.

        Parameters
        ----------
        k
            exchange rate [Hz]
        f
            pool size fraction
        dw
            chemical shift from water [ppm]
        lineshape
            lineshape of the MT pool ("Lorentzian", "SuperLorentzian")
        r1
            R1 relaxation rate [Hz] (1/T1)
            either r1 or t1 must be given, but not both
        t1
            T1 relaxation time [s] (1/R1)
            either t1 or r1 must be given, but not both
        r2
            R2 relaxation rate [Hz] (1/T2)
            either r2 or t2 must be given, but not both
        t2
            T2 relaxation time [s] (1/R2)
            either t2 or r2 must be given, but not both
        """
        super().__init__(f=f, dw=dw, r1=r1, r2=r2, t1=t1, t2=t2)

        self.k = k
        self.lineshape = lineshape

    def __dict__(self):
        """Return dictionary representation of MTPool."""
        return {
            'f': self.f,
            'r1': self.r1,
            'r2': self.r2,
            'k': self.k,
            'dw': self.dw,
            'lineshape': self.lineshape,
        }

    @property
    def k(self) -> float:
        """Exchange rate [Hz]."""
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        value = float(value)
        if value < 0:
            raise ValueError('k must be positive.')
        self._k = value

    @property
    def lineshape(self) -> str:
        """Lineshape of the MT pool."""
        return self._lineshape

    @lineshape.setter
    def lineshape(self, value: str) -> None:
        if value.lower() not in self.valid_lineshapes:
            raise ValueError('lineshape must be one of {self.valid_lineshapes}.')
        self._lineshape = value.lower()
