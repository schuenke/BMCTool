"""Definition of WaterPool class."""

from __future__ import annotations

from bmctool.parameters._Pool import Pool


class WaterPool(Pool):
    """Class to store WaterPool parameters."""

    __slots__ = ['_f', '_dw', '_r1', '_r2']

    def __init__(
        self,
        f: float = 1,
        r1: float = None,
        r2: float = None,
        t1: float = None,
        t2: float = None,
    ):
        """Initialize WaterPool object.

        Parameters
        ----------
        r1
            R1 relaxation rate [Hz] (1/T1)
        r2
            R2 relaxation rate [Hz] (1/T2)
        f, optional
            pool size fraction, by default 1
        """

        super().__init__(f=f, dw=0, r1=r1, r2=r2, t1=t1, t2=t2)

    def __dict__(self):
        """Return dictionary representation of WaterPool."""
        return {
            'f': self.f,
            'r1': self.r1,
            'r2': self.r2,
        }

    @property
    def dw(self) -> float:
        """Return chemical shift of WaterPool."""
        return self._dw

    @dw.setter
    def dw(self, value: float) -> None:
        raise UserWarning('Cannot set chemical shift of WaterPool. Value is fixed to 0.')
