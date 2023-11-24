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

    Parameters:
    -----------
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
    offsets : np.ndarray
        frequency offsets [ppm]
    m0_scan : np.ndarray
        # ToDo: add description and decide if it is needed
    """

    water_pool: WaterPool = dataclasses.field(default_factory=WaterPool)
    cest_pools: list = dataclasses.field(default_factory=list)
    mt_pool: MTPool = dataclasses.field(default_factory=MTPool)
    system: System = dataclasses.field(default_factory=System)
    options: Options = dataclasses.field(default_factory=Options)

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
    def from_yaml(cls, yaml_file: str | Path) -> Parameters:
        """Create a Params object from a yaml file.

        Parameters
        ----------
        yaml_file : str | Path
            Path to the yaml file.

        Returns
        -------
        Params
            Params object containing the simulation parameters.

        Raises
        ------
        FileNotFoundError
            If the yaml_file is not found.
        """
        if not Path(yaml_file).exists():
            raise FileNotFoundError(f'File {yaml_file} not found.')

        with open(yaml_file) as file:
            config = yaml.safe_load(file)

        # rename config keys to match the dataclass attributes according to rename dict
        rename = {'b1_inhomogeneity': 'b0_inhom', 'relb1': 'rel_b1'}

        sys_keys = System.__annotations__.keys()
        opt_keys = Options.__annotations__.keys()

        water_pool = WaterPool(**config['water_pool'])
        cest_pools = [CESTPool(**pool) for pool in config.get('cest_pools', [])]
        mt_pool = MTPool(**config.get('mt_pool', {})) if config.get('mt_pool') else None
        system = System(**{rename.get(key, key): value for key, value in config.items() if key in sys_keys})
        options = Options(**{rename.get(key, key): value for key, value in config.items() if key in opt_keys})

        return cls(water_pool, cest_pools, mt_pool, system, options)

    def set_water_pool(self, r1: float, r2: float, f: float = 1) -> WaterPool:
        """Set all water pool parameters.

        Parameters
        ----------
        r1 : float
            R1 relaxation rate [Hz] (1/T1)
        r2 : float
            R2 relaxation rate [Hz] (1/T2)
        f : float, optional
            Pool size fraction, by default 1

        Returns
        -------
        WaterPool
            WaterPool object containing the water pool parameters
        """

        water_pool = WaterPool(r1, r2, f)
        self.water_pool = water_pool

    def update_water_pool(self, **kwargs) -> WaterPool:
        """Update water pool parameters (r1, r2, f) given as kwargs.

        Returns
        -------
        WaterPool
            WaterPool object containing the updated water pool parameters

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """

        water_pool = WaterPool(**kwargs)
        self.water_pool = water_pool

    def add_cest_pool(self, r1: float, r2: float, k: float, f: float, dw: float) -> CESTPool:
        """Add a new CESTPool.

        Parameters
        ----------
        r1 : float
            R1 relaxation rate [Hz] (1/T1)
        r2 : float
            R2 relaxation rate [Hz] (1/T2)
        k : float
            exchange rate [Hz]
        f : float
            pool size fraction
        dw : float
            chemical shift from water [ppm]
        """

        cest_pool = CESTPool(r1, r2, k, f, dw)
        self.mz_loc += 2
        self.cest_pools.append(cest_pool)

    def update_cest_pool(self, pool_idx: int, **kwargs) -> CESTPool:
        """Update CEST pool parameters (r1, r2, k, f, dw) given as kwargs for a
        given pool.

        Parameters
        ----------
        pool_idx : int
            Index of the CEST pool to be updated
        kwargs : dict
            Parameters to be updated

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """

        cest_pool = CESTPool(**kwargs)
        self.cest_pools[pool_idx] = cest_pool

    def set_mt_pool(self, r1: float, r2: float, k: float, f: float, dw: float, lineshape: str) -> MTPool:
        """Set all MT pool parameters.

        Parameters
        ----------
        r1 : float
            R1 relaxation rate [Hz] (1/T1)
        r2 : float
            R2 relaxation rate [Hz] (1/T2)
        k : float
            exchange rate [Hz]
        f : float
            pool size fraction
        dw : float
            chemical shift from water [ppm]
        lineshape : str
            Lineshape of the MT pool ("Lorentzian", "SuperLorentzian" or "None")
        """

        mt_pool = MTPool(r1, r2, k, f, dw, lineshape)
        self.mt_pool = mt_pool

    def update_mt_pool(self, **kwargs) -> MTPool:
        """Update MT pool parameters (r1, r2, k, f, dw, lineshape) given as
        kwargs.

        Parameters
        ----------
        kwargs : dict
            Parameters to be updated

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """

        mt_pool = MTPool(**kwargs)
        self.mt_pool = mt_pool

    def set_scanner(self, b0: float, gamma: float, b0_inhom: float = 0, rel_b1: float = 1) -> System:
        """Set all scanner parameters.

        Parameters
        ----------
        b0 : float
            field strength [T]
        gamma : float
            gyromagnetic ratio [MHz/T]
        b0_inhom : float, optional
            B0 inhomogeneity [ppm], by default 0
        rel_b1 : float, optional
            B1 field scaling factor, by default 1
        """

        scanner = System(b0, gamma, b0_inhom, rel_b1)
        self.scanner = scanner

    def update_scanner(self, **kwargs) -> System:
        """Updates scanner values (kwargs: b0, gamma, b0_inhom, rel_b1)

        Parameters
        ----------
        kwargs : dict
            Parameters to be updated

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """

        scanner = System(**kwargs)
        self.scanner = scanner

    def set_options(
        self,
        verbose: bool = False,
        reset_init_mag: bool = True,
        scale: float = 1.0,
        max_pulse_samples: int = 500,
    ) -> Options:
        """Set all additional options.

        Parameters
        ----------
        verbose : bool, optional
            Flag to activate detailed outputs, by default False
        reset_init_mag : bool, optional
            flag to reset the initial magnetization for every offset, by default True
        scale : float, optional
            value of initial magnetization if reset_init_mag is True, by default 1.0
        max_pulse_samples : int, optional
            maximum number of simulation steps for one RF pulse, by default 500
        """

        options = Options(verbose, reset_init_mag, scale, max_pulse_samples)
        self.options = options

    def update_options(self, **kwargs) -> Options:
        """Update additional options (kwargs: verbose, reset_init_mag, scale,
        max_pulse_samples)

        Parameters
        ----------
        kwargs : dict
            Parameters to be updated

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """

        options = Options(**kwargs)
        self.options = options

    def _set_m_vec(self) -> np.ndarray:
        """Sets the initial magnetization vector (fully relaxed).

        Returns
        -------
        np.ndarray
            Initial magnetization vector (fully relaxed)

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
