"""Definition of System dataclass."""

from __future__ import annotations

import operator


class System:
    """Class to store system parameters."""

    __slots__ = ['_b0', '_gamma', '_b0_inhom', '_rel_b1']

    def __init__(
        self,
        b0: float,
        gamma: float,
        b0_inhom: float,
        rel_b1: float,
    ):
        """Initialize System instance.

        Parameters
        ----------
        b0
            field strength [T]
        gamma
            gyromagnetic ratio [MHz/T]
        b0_inhom
            B0 field inhomogeneity [ppm]
        rel_b1
            B1 field scaling factor
        """
        # set attributes using setters to check validity
        self.b0 = b0
        self.gamma = gamma
        self.b0_inhom = b0_inhom
        self.rel_b1 = rel_b1

    def __eq__(self, other: object) -> bool:
        """Check if two System instances are equal."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.__slots__ == other.__slots__:
            attr_getters = [operator.attrgetter(attr) for attr in self.__slots__]
            return all(getter(self) == getter(other) for getter in attr_getters)
        return False

    def __dict__(self):
        """Return dictionary representation of System."""
        return {
            'b0': self.b0,
            'gamma': self.gamma,
            'b0_inhom': self.b0_inhom,
            'rel_b1': self.rel_b1,
        }

    @property
    def b0(self) -> float:
        """Return field strength."""
        return self._b0

    @b0.setter
    def b0(self, value: float) -> None:
        value = float(value)
        if value < 0:
            raise ValueError('Field strength must be positive.')
        self._b0 = value

    @property
    def gamma(self) -> float:
        """Return gyromagnetic ratio."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = float(value)

    @property
    def b0_inhom(self) -> float:
        """Return B0 field inhomogeneity."""
        return self._b0_inhom

    @b0_inhom.setter
    def b0_inhom(self, value: float) -> None:
        self._b0_inhom = float(value)

    @property
    def rel_b1(self) -> float:
        """Return B1 field scaling factor."""
        return self._rel_b1

    @rel_b1.setter
    def rel_b1(self, value: float) -> None:
        value = float(value)
        if value < 0:
            raise ValueError('B1 field scaling factor must be positive.')
        self._rel_b1 = value
