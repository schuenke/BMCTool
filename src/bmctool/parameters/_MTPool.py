"""Definition of MTPool class."""

from __future__ import annotations

from bmctool.parameters._Pool import Pool


class MTPool(Pool):
    """Class to store MTPool parameters."""

    valid_lineshapes = ['lorentzian', 'superlorentzian']
    __slots__ = ['_r1', '_r2', '_k', '_f', '_dw', '_lineshape']

    def __init__(
        self,
        k: float,
        f: float,
        dw: float,
        lineshape: str,
        r1: float = None,
        r2: float = None,
        t1: float = None,
        t2: float = None,
    ):
        """Initialize MTPool object.

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
        lineshape
            lineshape of the MT pool ("Lorentzian", "SuperLorentzian")
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
