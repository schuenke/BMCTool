"""params.py Definition of the Params class, which stores all simulation
parameters."""
import numpy as np


class Params:
    """Class to store simulation parameters."""

    def __init__(self) -> None:
        """__init__ Initialize the Params class."""
        self.water_pool = {}
        self.cest_pools = []
        self.mt_pool = {}
        self.scanner = {}
        self.options = {}
        self.mz_loc = 0
        self.m_vec = None
        self.offsets = None
        self.m0_scan = None
        self.set_options()

    def set_water_pool(self, r1: float, r2: float, f: float = 1) -> dict:
        """set_water_pool Set all water pool parameters.

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
        dict
            Dictionary containing the water pool parameters
        """
        water_pool = {'r1': r1, 'r2': r2, 'f': f}
        self.water_pool = water_pool
        self.mz_loc += 2
        return water_pool

    def update_water_pool(self, **kwargs) -> dict:
        """update_water_pool Update water pool parameters (r1, r2, f) given as
        kwargs.

        Returns
        -------
        dict
            Dictionary containing the updated water pool parameters

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """
        option_names = ['r1', 'r2', 'f']
        if not all(name in option_names for name in kwargs.keys()):
            raise AttributeError('Unknown option name. Update aborted!')

        water_pool = {k: v for k, v in kwargs.items()}
        self.water_pool.update(water_pool)
        return water_pool

    def set_cest_pool(self, r1: float, r2: float, k: float, f: float, dw: float) -> dict:
        """set_cest_pool Set all CEST pool parameters.

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

        Returns
        -------
        dict
            Dictionary containing the CEST pool parameters
        """
        cest_pool = {'r1': r1, 'r2': r2, 'k': k, 'f': f, 'dw': dw}
        self.cest_pools.append(cest_pool)
        self.mz_loc += 2
        return cest_pool

    def update_cest_pool(self, pool_idx: int = 0, **kwargs) -> dict:
        """update_cest_pool Update CEST pool parameters (r1, r2, k, f, dw)
        given as kwargs for a given pool.

        Parameters
        ----------
        pool_idx : int, optional
            Index of the CEST pool to be updated, by default 0

        Returns
        -------
        dict
            Dictionary containing the updated CEST pool parameters

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """
        try:
            self.cest_pools[pool_idx]
        except IndexError:
            print(f"CEST pool # {pool_idx} doesn't exist. No parameters have been changed.")
            return

        option_names = ['r1', 'r2', 'k', 'f', 'dw']
        if not all(name in option_names for name in kwargs):
            raise AttributeError('Unknown option name. Update aborted!')

        cest_pool = {k: v for k, v in kwargs.items()}
        self.cest_pools[pool_idx].update(cest_pool)
        return cest_pool

    def set_mt_pool(self, r1: float, r2: float, k: float, f: float, dw: float, lineshape: str) -> dict:
        """set_mt_pool Set all MT pool parameters.

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

        Returns
        -------
        dict
            Dictionary containing the MT pool parameters
        """
        mt_pool = {'r1': r1, 'r2': r2, 'k': k, 'f': f, 'dw': dw, 'lineshape': lineshape}
        self.mt_pool.update(mt_pool)
        return mt_pool

    def update_mt_pool(self, **kwargs) -> dict:
        """update_mt_pool Update MT pool parameters (r1, r2, k, f, dw,
        lineshape) given as kwargs.

        Returns
        -------
        dict
            Dictionary containing the updated MT pool parameters

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """
        option_names = ['r1', 'r2', 'k', 'f', 'dw', 'lineshape']
        if not all(name in option_names for name in kwargs):
            raise AttributeError('Unknown option name. Update aborted!')

        mt_pool = {k: v for k, v in kwargs.items()}
        self.mt_pool.update(mt_pool)
        return mt_pool

    def set_scanner(self, b0: float, gamma: float, b0_inhom: float, rel_b1: float) -> dict:
        """set_scanner Set all scanner parameters.

        Parameters
        ----------
        b0 : float
            field strength [T]
        gamma : float
            gyromagnetic ratio [rad/uT]
        b0_inhom : float
            _description_
        rel_b1 : float
            _description_

        Returns
        -------
        dict
            _description_
        """
        scanner = {'b0': b0, 'gamma': gamma, 'b0_inhomogeneity': b0_inhom, 'rel_b1': rel_b1}
        self.scanner.update(scanner)
        return scanner

    def update_scanner(self, **kwargs) -> dict:
        """Updates scanner values (kwargs: b0, gamma, b0_inhomogeneity, rel_b1)
        :param kwargs: parameters to update the scanner options with, from b0,
        gamma, b0_inhomogeneity, rel_b1 :param b0: field strength [T] :param
        gamma: gyromagnetic ratio [rad/uT] :param b0_inhomogeneity: field
        ihnomogeneity [ppm] :param rel_b1: relative B1 field :return: dict
        containing the new parameter values."""
        kwargs = {('b0_inhomogeneity' if k == 'b0_inhom' else k): v for k, v in kwargs.items()}
        option_names = ['b0', 'gamma', 'b0_inhomogeneity', 'rel_b1']
        if not all(name in option_names for name in kwargs):
            raise AttributeError('Unknown option name. Update aborted!')

        scanner = {k: v for k, v in kwargs.items()}
        self.scanner.update(scanner)
        return scanner

    def _check_reset_init_mag(self) -> None:
        """_check_reset_init_mag Check if reset_init_mag and par_calc are
        compatible."""
        if not self.options['reset_init_mag'] and self.options['par_calc']:
            print("'par_calc = True' not available for 'reset_init_mag = False'. Changed 'par_calc' to 'False'.")
            self.options['par_calc'] = False

    def set_options(
        self,
        verbose: bool = False,
        reset_init_mag: bool = True,
        scale: float = 1.0,
        max_pulse_samples: int = 500,
        par_calc: bool = False,
    ) -> dict:
        """set_options Set all additional options.

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
        par_calc : bool, optional
            DEPRECATED

        Returns
        -------
        dict
            Dictionary containing the additional options
        """
        options = {
            'verbose': verbose,
            'reset_init_mag': reset_init_mag,
            'scale': scale,
            'max_pulse_samples': max_pulse_samples,
            'par_calc': par_calc,
        }
        self.options.update(options)
        self._check_reset_init_mag()
        return options

    def update_options(self, **kwargs) -> dict:
        """update_options Update additional options (kwargs: verbose,
        reset_init_mag, scale, max_pulse_samples, par_calc)

        Returns
        -------
        dict
            Dictionary containing the updated additional options

        Raises
        ------
        AttributeError
            If an unknown parameter is given
        """
        option_names = ['verbose', 'reset_init_mag', 'scale', 'max_pulse_samples', 'par_calc']

        if not all(name in option_names for name in kwargs):
            raise AttributeError('Unknown option name. Update aborted!')

        options = {k: v for k, v in kwargs.items()}
        self.options.update(options)
        self._check_reset_init_mag()
        return options

    def set_m_vec(self) -> np.ndarray:
        """set_m_vec Sets the initial magnetization vector (fully relaxed) from
        the defined pools with A: water pool, B: 1st CEST pool, C: MT pool, D:
        2nd CEST pool, etc. with possible inclusion of more CEST pools.

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
        m_vec[n_total_pools * 2] = self.water_pool['f']
        if self.cest_pools:
            for ii in range(1, n_total_pools):
                m_vec[n_total_pools * 2 + ii] = self.cest_pools[ii - 1]['f']
                m_vec[n_total_pools * 2] = m_vec[n_total_pools * 2]
        if self.mt_pool:
            m_vec = np.append(m_vec, self.mt_pool['f'])
        if isinstance(self.options['scale'], float):
            m_vec = m_vec * self.options['scale']

        self.m_vec = m_vec
        return m_vec

    def print_settings(self) -> None:
        """print_settings Prints the current parameters."""
        print('\n Current parameter settings:')
        print('\t water pool:\n', self.water_pool)
        print('\t CEST pools: \n', self.cest_pools)
        print('\t MT pool:\n', self.mt_pool)
        print('\t Scanner:\n', self.scanner)
        print('\t Options:\n', self.options)
