"""Definition of Pool base class."""

from __future__ import annotations

import operator
from abc import ABC


class Pool(ABC):
    """Base Class for Pools."""

    __slots__ = ['_r1', '_r2', '_f', '_dw']

    def __init__(
        self,
        f: float,
        dw: float,
        r1: float = None,
        r2: float = None,
        t1: float = None,
        t2: float = None,
    ):
        """Init method for abstract Pool class.

        Parameters
        ----------
        r1
            R1 relaxation rate [Hz] (1/T1)
        r2
            R2 relaxation rate [Hz] (1/T2)
        f
            pool size fraction
        dw
            chemical shift from water [ppm]
        """

        if (r1 is None) == (t1 is None):
            raise ValueError('Either r1 or t1 must be given, but not both.')

        if (r2 is None) == (t2 is None):
            raise ValueError('Either r2 or t2 must be given, but not both.')

        if r1 is not None:
            self.r1 = r1
        elif t1 is not None:
            self.t1 = t1

        if r2 is not None:
            self.r2 = r2
        elif t2 is not None:
            self.t2 = t2

        self.f = f
        self.dw = dw

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.__slots__ == other.__slots__:
                attr_getters = [operator.attrgetter(attr) for attr in self.__slots__]
                return all(getter(self) == getter(other) for getter in attr_getters)
        return False

    @property
    def r1(self) -> float:
        """R1 relaxation rate [Hz] (1/T1)."""
        return self._r1

    @r1.setter
    def r1(self, value: float) -> None:
        value = float(value)
        if value < 0:
            raise ValueError('r1 must be positive.')
        self._r1 = value

    @property
    def r2(self) -> float:
        """R2 relaxation rate [Hz] (1/T2)."""
        return self._r2

    @r2.setter
    def r2(self, value: float) -> None:
        value = float(value)
        if value < 0:
            raise ValueError('r2 must be positive.')
        self._r2 = value

    @property
    def f(self) -> float:
        """Pool size fraction."""
        return self._f

    @f.setter
    def f(self, value: float) -> None:
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError('f must be between 0 and 1.')
        self._f = value

    @property
    def dw(self) -> float:
        """Chemical shift from water [ppm]."""
        return self._dw

    @dw.setter
    def dw(self, value: float) -> None:
        self._dw = float(value)

    @property
    def t1(self) -> float:
        """T1 relaxation time [s]."""
        return 1 / self._r1

    @t1.setter
    def t1(self, value: float) -> None:
        self.r1 = 1 / float(value)

    @property
    def t2(self) -> float:
        """T2 relaxation time [s]."""
        return 1 / self._r2

    @t2.setter
    def t2(self, value: float) -> None:
        self.r2 = 1 / float(value)

    @classmethod
    def from_dict(cls, data: dict) -> Pool:
        """Create Pool instance from dictionary."""
        return cls(**data)
