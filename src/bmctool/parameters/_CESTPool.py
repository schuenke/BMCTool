"""Definition of CESTPool class."""

from __future__ import annotations

from bmctool.parameters._Pool import Pool


class CESTPool(Pool):
    """Class to store CESTPool parameters."""

    __slots__ = ['_r1', '_r2', '_k', '_f', '_dw']

    def __init__(
        self,
        k: float,
        f: float,
        dw: float,
        r1: float = None,
        r2: float = None,
        t1: float = None,
        t2: float = None,
    ):
        """Initialize CESTPool object.

        Parameters
        ----------
        r1
            R1 relaxation rate [Hz] (1/T1)
        r2
            R2 relaxation rate [Hz] (1/T2)
        k
            exchange rate [Hz]
        f
            pool size fraction
        dw
            chemical shift from water [ppm]
        """

        super().__init__(f=f, dw=dw, r1=r1, r2=r2, t1=t1, t2=t2)
        self.k = k

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
