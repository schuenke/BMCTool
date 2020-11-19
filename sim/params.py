"""
params.py
    Class definition to store simulation parameters
"""
import numpy as np
from sim.utils.utils import check_m0_scan, get_offsets
from sim.utils.seq.read import read_any_version


class Params:
    """
    Class to store simulation parameters
    """
    def __init__(self, set_defaults: bool = False):
        """
        :param set_defaults: if True, class initializes with parameters for a standard APT weightedCEST simulation of
        gray matter at 3T with an amide CEST pool, a creatine CEST pool and a Lorentzian shaped MT pool
        """
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

    def set_water_pool(self,
                       r1: float,
                       r2: float,
                       f: float = 1) \
            -> dict:
        """
        defining the water pool for simulation
        :param r1: relaxation rate R1 = 1/ T1 [Hz]
        :param r2: relaxation rate R2 = 1/T2 [Hz]
        :param f: proton fraction (default = 1)
        :return: dict of water pool parameters
        """
        water_pool = {'r1': r1, 'r2': r2, 'f': f}
        self.water_pool = water_pool
        self.mz_loc += 2
        return water_pool

    def update_water_pool(self, **kwargs) -> dict:
        """
        Updates water settings (kwargs: r1, r2, f)
        kwargs: parameter to update the water pool with, from r1, r2, f
        :param r1: relaxation rate R1 = 1/ T1 [Hz]
        :param r2: relaxation rate R2 = 1/T2 [Hz]
        :param f: proton fraction (default = 1)
        :return: dict containing the new water_pool settings
        """
        option_names = ['r1', 'r2', 'f']
        if not all(name in option_names for name in kwargs.keys()):
            raise AttributeError('Unknown option name. Update aborted!')

        water_pool = {k: v for k, v in kwargs.items()}
        self.water_pool.update(water_pool)
        return water_pool

    def set_cest_pool(self,
                      r1: float,
                      r2: float,
                      k: int,
                      f: float,
                      dw: float) \
            -> dict:
        """
        defines a CEST pool for simulation
        :param r1: relaxation rate R1 = 1/T1 [Hz]
        :param r2: relaxation rate R2 = 1/T2 [Hz]
        :param k: exchange rate [Hz]
        :param f: proton fraction
        :param dw: chemical shift from water [ppm]
        :return: dict of CEST pool parameters
        """
        cest_pool = {'r1': r1, 'r2': r2, 'k': k, 'f': f, 'dw': dw}
        self.cest_pools.append(cest_pool)
        self.mz_loc += 2
        return cest_pool

    def update_cest_pool(self, pool_idx: int = 0, **kwargs) -> dict:
        """
        Updates CEST pool values (kwargs: r1, r2, k, f, dw)
        :param pool_idx: index of the CEST pool that should be changed
        :param kwargs: parameter(s) to update the defined CEST pool with, from r1, r2, k, f, dw
        :param r1: relaxation rate R1 = 1/ T1 [Hz]
        :param k: exchange rate [Hz]
        :param r2: relaxation rate R2 = 1/T2 [Hz]
        :param f: proton fraction (default = 1)
        :param dw: chemical shift from water [ppm]
        :return: dict of the new CEST pool parameters
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

    def set_mt_pool(self,
                    r1: float,
                    r2: float,
                    k: int,
                    f: float,
                    dw: int,
                    lineshape: str) \
            -> dict:
        """
        defines an MT pool for simulation
        :param r1: relaxation rate R1 = 1/ T1 [Hz]
        :param r2: relaxation rate R2 = 1/ T2 [Hz]
        :param k: exchange rate [Hz]
        :param f: proton fraction
        :param dw: chemical shift from water [ppm]
        :param lineshape: shape of MT pool ("Lorentzian", "SuperLorentzian" or "None")
        :return: dict containing MT pool parameters
        """
        mt_pool = {'r1': r1, 'r2': r2, 'k': k, 'f': f, 'dw': dw, 'lineshape': lineshape}
        self.mt_pool.update(mt_pool)
        return mt_pool

    def update_mt_pool(self, **kwargs) -> dict:
        """
        Updates mt pool values (kwargs: r1, r2, k, f, dw)
        :param kwargs: parameter(s) to update the MT pool with, from r1, r2, k, f, dw
        :param r1: relaxation rate R1 = 1/ T1 [Hz]
        :param r2: relaxation rate R2 = 1/ T2 [Hz]
        :param k: exchange rate [Hz]
        :param f: proton fraction
        :param dw: chemical shift from water [ppm]
        :param lineshape: shape of MT pool ("Lorentzian", "SuperLorentzian" or "None")
        :return: dict containing the new MT pool parameters
        """
        option_names = ['r1', 'r2', 'k', 'f', 'dw', 'lineshape']
        if not all(name in option_names for name in kwargs):
            raise AttributeError('Unknown option name. Update aborted!')

        mt_pool = {k: v for k, v in kwargs.items()}
        self.mt_pool.update(mt_pool)
        return mt_pool

    def set_scanner(self,
                    b0: float,
                    gamma: float,
                    b0_inhom: float,
                    rel_b1: float) \
            -> dict:
        """
        Sets the scanner values
        :param b0: field strength [T]
        :param gamma: gyromagnetic ratio [rad/uT]
        :param b0_inhom: field ihnomogeneity [ppm]
        :param rel_b1: relative B1 field
        :return: dict containing the parameter values
        """
        scanner = {'b0': b0, 'gamma': gamma, 'b0_inhomogeneity': b0_inhom, 'rel_b1': rel_b1}
        self.scanner.update(scanner)
        return scanner

    def update_scanner(self, **kwargs) -> dict:
        """
        Updates scanner values (kwargs: b0, gamma, b0_inhomogeneity, rel_b1)
        :param kwargs: parameters to update the scanner options with, from b0, gamma, b0_inhomogeneity, rel_b1
        :param b0: field strength [T]
        :param gamma: gyromagnetic ratio [rad/uT]
        :param b0_inhomogeneity: field ihnomogeneity [ppm]
        :param rel_b1: relative B1 field
        :return: dict containing the new parameter values
        """
        kwargs = {('b0_inhomogeneity' if k == 'b0_inhom' else k): v for k, v in kwargs.items()}
        option_names = ['b0', 'gamma', 'b0_inhomogeneity', 'rel_b1']
        if not all(name in option_names for name in kwargs):
            raise AttributeError('Unknown option name. Update aborted!')

        scanner = {k: v for k, v in kwargs.items()}
        self.scanner.update(scanner)
        return scanner

    def _check_reset_init_mag(self):
        if not self.options['reset_init_mag'] and self.options['par_calc']:
            print(f"'par_calc = True' not available for 'reset_init_mag = False'. Changed 'par_calc' to 'False'.")
            self.options['par_calc'] = False

    def set_options(self,
                    verbose: bool = False,
                    reset_init_mag: bool = True,
                    scale: float = 1.0,
                    max_pulse_samples: int = 500,
                    par_calc: bool = True) \
            -> dict:
        """
        Setting additional options
        :param verbose: Verbose output
        :param reset_init_mag: true if magnetization should be set to self.m_vec after each ADC
        :param scale: scaling factor for the magnetization after reset (if reset_init_mag = True)
        :param max_pulse_samples: max number of samples for shaped pulses
        :param par_calc: toggles parallel calculation (BMCTool only)
        :return: dict containing option parameters
        """
        options = {'verbose': verbose,
                   'reset_init_mag': reset_init_mag,
                   'scale': scale,
                   'max_pulse_samples': max_pulse_samples,
                   'par_calc': par_calc}
        self.options.update(options)
        self._check_reset_init_mag()
        return options

    def update_options(self, **kwargs) -> dict:
        """
        Updates additional options (kwargs: verbose, reset_init_mag, scale, max_pulse_samples, par_calc)
        :param kwargs: parameters to update the options with from verbose, reset_init_mag, scale, max_pulse_samples,
                    par_calc
        :param verbose: Verbose output
        :param reset_init_mag: true if magnetization should be set to self.m_vec after each ADC
        :param scale: scaling factor for the magnetization after reset (if reset_init_mag = True)
        :param max_pulse_samples: max number of samples for shaped pulses
        :param par_calc: toggles parallel calculation (BMCTool only)
        :return: dict containing the new option parameters
        """
        option_names = ['verbose', 'reset_init_mag', 'scale', 'max_pulse_samples', 'par_calc']

        if not all(name in option_names for name in kwargs):
            raise AttributeError('Unknown option name. Update aborted!')

        options = {k: v for k, v in kwargs.items()}
        self.options.update(options)
        self._check_reset_init_mag()
        return options

    def set_m_vec(self) -> np.array:
        """
        Sets the initial magnetization vector (fully relaxed) from the defined pools
        e. g. for 2 CEST pools: [MxA, MxB, MxD, MyA, MyB, MyD, MzA, MzB, MzD, MzC]
        with A: water pool, B: 1st CEST pool, D: 2nd CEST pool, C: MT pool
        with possible inclusion of more CEST pools in the same way

        :return: array of the initial magnetizations
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
        if type(self.options['scale']) == int or float:
            m_vec = m_vec * self.options['scale']

        self.m_vec = m_vec
        return m_vec

    def print_settings(self):
        """
        function to print the current parameter settings
        """
        print("\n Current parameter settings:")
        print("\t water pool:\n", self.water_pool)
        print("\t CEST pools: \n", self.cest_pools)
        print("\t MT pool:\n", self.mt_pool)
        print("\t Scanner:\n", self.scanner)
        print("\t Options:\n", self.options)

    def set_definitions(self, seq_file: str) -> [np.ndarray, bool]:
        """
        saves the definitions from the sequence file to the Params object
        :param seq_file: path to the sequence file
        :return (offsets, m0_scan): the offsets and m0_scan values defined in the sequence file
        """
        seq = read_any_version(seq_file)
        self.offsets = get_offsets(seq)
        self.m0_scan = check_m0_scan(seq)
        return self.offsets, self.m0_scan

    def get_num_adc_events(self) -> int:
        """
        :return num_adc_events: number of ADC events based on the number of offsets
        """
        num_adc_events = len(self.offsets)
        return num_adc_events
