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
                       r1: float = None,
                       r2: float = None,
                       f: float = 1) \
            -> dict:
        """
        defining the water pool for simulation
        :param r1: relaxation rate R1 = 1/ T1 [Hz]
        :param r2: relaxation rate R2 = 1/T2 [Hz]
        :param f: proton fraction (default = 1)
        :return: dict of water pool parameters
        """
        if None in [r1, r2]:
            raise ValueError('Not enough parameters given for water pool definition.')

        water_pool = {'r1': r1, 'r2': r2, 'f': f}
        self.water_pool = water_pool
        self.mz_loc += 2
        return water_pool

    def update_water_pool(self, water_dict: dict = {}) -> dict:
        """
        Updates water settings
        :param water_dict: dict with items that should be updated
        :return:
        """
        option_names = ['r1', 'r2', 'f']
        if not all(name in option_names for name in water_dict):
            raise AttributeError('Unknown option name. Update aborted!')

        water_pool = {k: v for k, v in water_dict.items()}
        self.water_pool.update(water_pool)
        return water_pool

    def set_cest_pool(self,
                      r1: float = None,
                      r2: float = None,
                      k: int = None,
                      f: float = None,
                      dw: float = None) \
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
        if None in [r1, r2, k, f, dw]:
            raise ValueError('Not enough parameters given for CEST pool definition.')

        cest_pool = {'r1': r1, 'r2': r2, 'k': k, 'f': f, 'dw': dw}
        self.cest_pools.append(cest_pool)
        self.mz_loc += 2
        return cest_pool

    def update_cest_pool(self, pool_num: int = 1, cest_dict: dict = {}) -> dict:
        """
        Updates mt pool values
        :param pool_num: number of the CEST pool that should be changed
        :param mt_dict: dict with items that should be updated
        :return:
        """
        try:
            old_dict = self.cest_pools[pool_num]
        except IndexError:
            print(f"CEST pool # {pool_num} doesn't exist. No parameters have been changed.")
            return

        option_names = ['r1', 'r2', 'k', 'f', 'dw']
        if not all(name in option_names for name in cest_dict):
            raise AttributeError('Unknown option name. Update aborted!')

        cest_pool = {k: v for k, v in cest_dict.items()}
        self.cest_pools[pool_num].update(cest_pool)
        return cest_pool

    def set_mt_pool(self,
                    r1: float = None,
                    r2: float = None,
                    k: int = None,
                    f: float = None,
                    dw: int = None,
                    lineshape: str = None) \
            -> dict:
        """
        defines an MT pool for simulation
        :param r1: relaxation rate R1 = 1/ T1 [Hz]
        :param r2: relaxation rate R2 = 1/ T2 [Hz]
        :param k: exchange rate [Hz]
        :param f: proton fraction
        :param dw: chemical shift from water [ppm]
        :param lineshape: shape of MT pool ("Lorentzian", "SuperLorentzian" or "None")
        :return:
        """
        if None in [r1, r2, k, f, dw, lineshape]:
            raise ValueError('Not enough parameters given for MT pool definition.')

        mt_pool = {'r1': r1, 'r2': r2, 'k': k, 'f': f, 'dw': dw, 'lineshape': lineshape}
        self.mt_pool.update(mt_pool)
        return mt_pool

    def update_mt_pool(self, mt_dict: dict = {}) -> dict:
        """
        Updates mt pool values
        :param mt_dict: dict with items that should be updated
        :return:
        """
        option_names = ['r1', 'r2', 'k', 'f', 'dw', 'lineshape']
        if not all(name in option_names for name in mt_dict):
            raise AttributeError('Unknown option name. Update aborted!')

        mt_pool = {k: v for k, v in mt_dict.items()}
        self.mt_pool.update(mt_pool)
        return mt_pool

    def set_scanner(self,
                    b0: float = None,
                    gamma: float = None,
                    b0_inhom: float = None,
                    rel_b1: float = None) \
            -> dict:
        """
        Sets the scanner values
        :param b0: field strength [T]
        :param gamma: gyromagnetic ratio [rad/uT]
        :param b0_inhom: field ihnomogeneity [ppm]
        :param rel_b1: relative B1 field
        :return: library containing the parameter values
        """
        if None in [b0, gamma, b0_inhom, rel_b1]:
            raise ValueError('Not enough parameters given for scanner definition.')

        scanner = {'b0': b0, 'gamma': gamma, 'b0_inhomogeneity': b0_inhom, 'rel_b1': rel_b1}
        self.scanner.update(scanner)
        return scanner

    def update_scanner(self, scanner_dict: dict = {}) -> dict:
        """
        Updates scanner values
        :param scanner_dict: dict with items that should be updated
        :return:
        """
        option_names = ['b0', 'gamma', 'b0_inhom', 'rel_b1']
        if not all(name in option_names for name in scanner_dict):
            raise AttributeError('Unknown option name. Update aborted!')

        scanner = {k: v for k, v in scanner_dict.items()}
        self.scanner.update(scanner)
        return scanner

    def set_options(self,
                    verbose: bool = False,
                    reset_init_mag: bool = True,
                    scale: float = 1.0,
                    max_pulse_samples: int = 500,
                    par_calc: bool = False) \
            -> dict:
        """
        Setting additional options
        :param verbose: Verbose output
        :param reset_init_mag: true if magnetization should be set to self.m_vec after each ADC
        :param scale: scaling factor for the magnetization after reset (if reset_init_mag = True)
        :param max_pulse_samples: max number of samples for shaped pulses
        :param par_calc: toggles parallel calculation (BMCTool only)
        :return:
        """
        options = {'verbose': verbose,
                   'reset_init_mag': reset_init_mag,
                   'scale': scale,
                   'max_pulse_samples': max_pulse_samples,
                   'par_calc': par_calc}
        self.options.update(options)
        return options

    def update_options(self, options_dict: dict = {}) -> dict:
        """
        Updates additional options
        :param options_dict: dict with items that should be updated
        :return:
        """
        option_names = ['verbose', 'reset_init_mag', 'scale', 'max_pulse_samples', 'par_calc']

        if not all(name in option_names for name in options_dict):
            raise AttributeError('Unknown option name. Update aborted!')

        options = {k: v for k, v in options_dict.items()}
        self.options.update(options)
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
