"""Definition of Options class."""

import operator

from bmctool.utils import truthy_check


class Options:
    """Class to store simulation options."""

    __slots__ = ['_verbose', '_reset_init_mag', '_scale', '_max_pulse_samples']

    def __init__(
        self,
        verbose: bool = False,
        reset_init_mag: bool = True,
        scale: float = 1.0,
        max_pulse_samples: int = 500,
    ):
        """Initialize Options instance.

        Parameters
        ----------
        verbose
            Print verbose output, default False.
        reset_init_mag
            Reset initial magnetization to "scale" value, default True.
        scale
            Scale factor for initial magnetization, default 1.0.
        max_pulse_samples
            Maximum number of samples in a pulse, default 500.
        """
        # set attributes using setters to check validity
        self.verbose = verbose
        self.reset_init_mag = reset_init_mag
        self.scale = scale
        self.max_pulse_samples = max_pulse_samples

    def __eq__(self, other: object) -> bool:
        """Check if two Options instances are equal."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.__slots__ == other.__slots__:
            attr_getters = [operator.attrgetter(attr) for attr in self.__slots__]
            return all(getter(self) == getter(other) for getter in attr_getters)
        return False

    def __dict__(self):
        """Return dictionary representation of Options."""
        return {
            'verbose': self.verbose,
            'reset_init_mag': self.reset_init_mag,
            'scale': self.scale,
            'max_pulse_samples': self.max_pulse_samples,
        }

    @property
    def verbose(self) -> bool:
        """Return verbose option."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = truthy_check(value)

    @property
    def reset_init_mag(self) -> bool:
        """Return reset_init_mag option."""
        return self._reset_init_mag

    @reset_init_mag.setter
    def reset_init_mag(self, value: bool) -> None:
        self._reset_init_mag = truthy_check(value)

    @property
    def scale(self) -> float:
        """Return scale option."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError('scale must be between 0 and 1.')
        self._scale = value

    @property
    def max_pulse_samples(self) -> int:
        """Return max_pulse_samples option."""
        return self._max_pulse_samples

    @max_pulse_samples.setter
    def max_pulse_samples(self, value: int) -> None:
        value = int(value)
        if value < 1:
            raise ValueError('max_pulse_samples must be positive.')
        self._max_pulse_samples = value

    @classmethod
    def from_dict(cls, dictionary: dict):
        """Create Options instance from dictionary."""
        return cls(**dictionary)
