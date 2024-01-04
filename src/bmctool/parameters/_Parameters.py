"""Definition of Params dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml

from bmctool.parameters import CESTPool
from bmctool.parameters import MTPool
from bmctool.parameters import Options
from bmctool.parameters import System
from bmctool.parameters import WaterPool


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

    water_pool: WaterPool = dataclasses.field(default_factory=WaterPool)
    cest_pools: list = dataclasses.field(default_factory=list)
    mt_pool: MTPool = dataclasses.field(default_factory=MTPool)
    system: System = dataclasses.field(default_factory=System)
    options: Options = dataclasses.field(default_factory=Options)

    def __eq__(self, other):
        if isinstance(other, Parameters):
            return (
                self.water_pool == other.water_pool
                and self.cest_pools == other.cest_pools
                and self.mt_pool == other.mt_pool
                and self.system == other.system
                and self.options == other.options
            )
        return False

    @property
    def num_cest_pools(self):
        """Get the number of pools."""
        return len(self.cest_pools) if self.cest_pools else 0

    @property
    def mz_loc(self):
        """Get the location of the water z-magnetization in the BMC matrix."""
        if not self.water_pool:
            raise Exception('No water pool defined. mz_loc cannot be determined')

        # mz_loc is 2 for water pool and +2 for each CEST pool
        return 2 + 2 * len(self.cest_pools)

    @property
    def m_vec(self):
        """Get the initial magnetization vector (fully relaxed)."""
        if not self.water_pool:
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
    def from_dict(cls, config: dict) -> Parameters:
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
        sys_keys = [
            attr for attr in System.__dict__ if not callable(getattr(System, attr)) and not attr.startswith('_')
        ]
        opt_keys = [
            attr for attr in Options.__dict__ if not callable(getattr(Options, attr)) and not attr.startswith('_')
        ]

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
    def from_yaml(cls, yaml_file: str | Path) -> Parameters:
        """Create a Parameters instance from a yaml config file.

        Parameters
        ----------
        yaml_file
            Path to yaml config file.
        """

        if not Path(yaml_file).exists():
            raise FileNotFoundError(f'File {yaml_file} not found.')

        with open(yaml_file) as file:
            config = yaml.safe_load(file)

        return cls.from_dict(config)

    def _set_m_vec(self) -> np.ndarray:
        """Set the initial magnetization vector (fully relaxed).

        Returns
        -------
        m_vec
            Initial magnetization vector (fully relaxed) as numpy array.

        Raises
        ------
        Exception
            If no water pool is defined.
        """

        if not self.water_pool:
            raise Exception('No water pool defined before assignment of magnetization vector.')

        if self.cest_pools:
            n_total_pools = len(self.cest_pools) + 1
        else:
            n_total_pools = 1

        m_vec = np.zeros(n_total_pools * 3)
        m_vec[n_total_pools * 2] = self.water_pool.f
        if self.cest_pools:
            for ii in range(1, n_total_pools):
                m_vec[n_total_pools * 2 + ii] = self.cest_pools[ii - 1].f
                m_vec[n_total_pools * 2] = m_vec[n_total_pools * 2]
        if self.mt_pool:
            m_vec = np.append(m_vec, self.mt_pool.f)
        if isinstance(self.options.scale, float):
            m_vec = m_vec * self.options.scale

        self.m_vec = m_vec
        return m_vec

    def to_yaml(self, yaml_file: str | Path) -> None:
        """Export parameters to yaml file.

        Parameters
        ----------
        yaml_file
            Path to yaml file.
        """

        if self.cest_pools:
            cest_dict = {f'cest_{ii + 1}': pool.__dict__() for ii, pool in enumerate(self.cest_pools)}

        with open(yaml_file, 'w') as file:
            yaml.dump({'water_pool': self.water_pool.__dict__()}, file)
            if self.mt_pool:
                yaml.dump({'mt_pool': self.mt_pool.__dict__()}, file)
            if self.cest_pools:
                yaml.dump({'cest_pool': cest_dict}, file)
            yaml.dump(self.system.__dict__(), file)
            yaml.dump(self.options.__dict__(), file)

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
