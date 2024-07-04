"""Definition of CESTPool class."""

from __future__ import annotations

from bmctool.parameters.Pool import Pool


class CESTPool(Pool):
    """Class to store CESTPool parameters."""

    __slots__ = ['_r1', '_r2', '_k', '_f', '_dw']

    def __init__(
        self,
        k: float,
        f: float,
        dw: float,
        r1: float | None = None,
        r2: float | None = None,
        t1: float | None = None,
        t2: float | None = None,
    ):
        """Initialize CESTPool object.

        Parameters
        ----------
        k
            exchange rate [Hz]
        f
            pool size fraction
        dw
            chemical shift from water [ppm]
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

    def __dict__(self):
        """Return dictionary representation of CESTPool."""
        return {
            'f': self.f,
            'r1': self.r1,
            'r2': self.r2,
            'k': self.k,
            'dw': self.dw,
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
