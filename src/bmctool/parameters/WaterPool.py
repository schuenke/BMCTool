"""Definition of WaterPool class."""

from bmctool.parameters.Pool import Pool


class WaterPool(Pool):
    """Class to store WaterPool parameters."""

    __slots__ = ['_f', '_dw', '_r1', '_r2']

    def __init__(
        self,
        f: float = 1,
        r1: float | None = None,
        r2: float | None = None,
        t1: float | None = None,
        t2: float | None = None,
    ):
        """Initialize WaterPool object.

        Parameters
        ----------
        f, optional
            pool size fraction, by default 1
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
        raise UserWarning(f'Cannot set chemical shift of WaterPool to {value}. Value is fixed to 0.')
