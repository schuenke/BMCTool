"""Definition of Params dataclass."""

import dataclasses
from pathlib import Path
from typing import Self

import numpy as np
import yaml

from bmctool.parameters.CESTPool import CESTPool
from bmctool.parameters.MTPool import MTPool
from bmctool.parameters.Options import Options
from bmctool.parameters.System import System
from bmctool.parameters.WaterPool import WaterPool


@dataclasses.dataclass(slots=True)
class Parameters:
    """Class to store simulation parameters.

    Parameters
    ----------
    water_pool : WaterPool
        Water pool parameters
    cest_pools : list
        List of CESTPool objects
    mt_pool : MTPool
        MT pool parameters
    system : System
        System parameters
    options : Options
        Additional options
    """

    water_pool: WaterPool
    cest_pools: list[CESTPool]
    mt_pool: MTPool | None
    system: System
    options: Options

    def __eq__(self, other: object) -> bool:
        """Check if two Parameters instances are equal."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        # Todo: check if cest_pool comparison is correct and add test(s)
        if self.__slots__ == other.__slots__:
            return (
                self.water_pool == other.water_pool
                and all(
                    self.cest_pools[ii] == other.cest_pools[ii] for ii in range(len(self.cest_pools)) if self.cest_pools
                )
                and self.mt_pool == other.mt_pool
                and self.system == other.system
                and self.options == other.options
            )
        return False

    @property
    def num_cest_pools(self) -> int:
        """Get the number of pools."""
        return len(self.cest_pools) if self.cest_pools else 0

    @property
    def mz_loc(self) -> int:
        """Get the location of the water z-magnetization in the BMC matrix."""
        if not hasattr(self, 'water_pool'):
            raise Exception('No water pool defined. mz_loc cannot be determined')

        # mz_loc is 2 for water pool and +2 for each CEST pool
        return 2 + 2 * len(self.cest_pools)

    def to_dict(self) -> dict:
        """Return dictionary representation with leading underscores removed."""
        return {slot.lstrip('_'): getattr(self, slot) for slot in self.__slots__}

    @property
    def m_vec(self) -> np.ndarray:
        """Get the initial magnetization vector (fully relaxed)."""
        if not hasattr(self, 'water_pool'):
            raise Exception('No water pool defined. m_vec cannot be determined')

        num_full_pools = self.num_cest_pools + 1

        m_vec = np.zeros(num_full_pools * 3)
        m_vec[num_full_pools * 2] = self.water_pool.f

        if self.cest_pools:
            m_vec[num_full_pools * 2 + 1 : num_full_pools * 3] = [pool.f for pool in self.cest_pools]

        if self.mt_pool:
            m_vec = np.append(m_vec, self.mt_pool.f)

        return m_vec * self.options.scale

    @classmethod
    def from_dict(cls, config: dict) -> Self:
        """Create a Parameters instance from a dictionary.

        Parameters
        ----------
        config
            Dictionary containing the simulation parameters.
        """
        # create a dictionary with name pairs for renaming to match class attributes
        rename = {
            'b0_inhomogeneity': 'b0_inhom',
            'relb1': 'rel_b1',
            'max_samples': 'max_pulse_samples',
        }

        #
        sys_keys = [attr.lstrip('_') for attr in System.__slots__]

        opt_keys = [attr.lstrip('_') for attr in Options.__slots__]

        water_pool = WaterPool(**config['water_pool'])
        cest_pools = [CESTPool(**pool) for pool in config.get('cest_pool', {}).values()]

        mt_pool = MTPool(**config.get('mt_pool', {})) if config.get('mt_pool') else None
        system = System(
            **{rename.get(key, key): value for key, value in config.items() if rename.get(key, key) in sys_keys}
        )

        options = Options(
            **{rename.get(key, key): value for key, value in config.items() if rename.get(key, key) in opt_keys}
        )

        return cls(water_pool, cest_pools, mt_pool, system, options)

    @classmethod
    def from_yaml(cls, yaml_file: str | Path) -> Self:
        """Create a Parameters instance from a yaml config file.

        Parameters
        ----------
        yaml_file
            Path to yaml config file.
        """
        with Path(yaml_file).open() as file:
            config = yaml.safe_load(file)

        return cls.from_dict(config)

    def to_yaml(self, yaml_file: str | Path) -> None:
        """Export parameters to yaml file.

        Parameters
        ----------
        yaml_file
            Path to yaml file.
        """
        if self.cest_pools:
            cest_dict = {f'cest_{ii + 1}': pool.to_dict() for ii, pool in enumerate(self.cest_pools)}

        with Path(yaml_file).open('w') as file:
            yaml.dump({'water_pool': self.water_pool.to_dict()}, file)
            if self.mt_pool:
                yaml.dump({'mt_pool': self.mt_pool.to_dict()}, file)
            if self.cest_pools:
                yaml.dump({'cest_pool': cest_dict}, file)
            yaml.dump(self.system.to_dict(), file)
            yaml.dump(self.options.to_dict(), file)

        file.close()

    def add_cest_pool(self, cest_pool: CESTPool) -> None:
        """Add a CESTPool object to the cest_pools list."""
        self.cest_pools.append(cest_pool)

    def update_cest_pool(self, idx: int, **kwargs) -> None:
        """Update parameters for CEST pool with index idx.

        Available parameters are: r1 or t1, r2 or t2, k, f, dw.

        Parameters
        ----------
        idx
            Index of the CEST pool to be updated
        kwargs
            Parameters to be updated (r1 or t1, r2 or t2, k, f, dw)
        """
        for key, value in kwargs.items():
            setattr(self.cest_pools[idx], key, value)

    def update_mt_pool(self, **kwargs) -> None:
        """Update MT pool parameters.

        Available parameters are: r1 or t1, r2 or t2, f, k, dw.
        """
        for key, value in kwargs.items():
            setattr(self.mt_pool, key, value)

    def update_options(self, **kwargs) -> None:
        """Update options parameters.

        Available parameters are: verbose, reset_init_mag, scale, max_pulse_samples.
        """
        for key, value in kwargs.items():
            setattr(self.options, key, value)

    def update_system(self, **kwargs) -> None:
        """Update scanner parameters.

        Available parameters are: b0, gamma, b0_inhom, rel_b1.
        """
        for key, value in kwargs.items():
            setattr(self.system, key, value)

    def update_water_pool(self, **kwargs) -> None:
        """Update water pool parameters.

        Available parameters are: r1 or t1, r2 or t2, f.
        """
        for key, value in kwargs.items():
            setattr(self.water_pool, key, value)
