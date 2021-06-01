"""
bmc_tool.py
    Tool to solve the Bloch-McConnell (BMC) equations using a (parallelized) eigenwert ansatz.
"""
import numpy as np
from typing import Union
from pathlib import Path
from tqdm import tqdm

from pypulseq.Sequence.read_seq import __strip_line as strip_line

from bmctool.params import Params
from bmctool.bmc_solver import BlochMcConnellSolver
from bmctool.utils.seq.read import read_any_version


class BMCTool:
    """
    Bloch-McConnell (BMC) simulation tool.
    :param params: Params object including all experimental and sample settings.
    :param seq_file: path of the *.seq file
    :param verbose: bool to deactivate print commands
    """
    def __init__(self, params: Params,
                 seq_file: Union[str, Path],
                 verbose: bool = True):
        self.params = params
        self.seq_file = seq_file
        self.par_calc = params.options['par_calc']
        self.verbose = verbose
        self.run_m0_scan = None
        self.bm_solver = None

        self.seq = read_any_version(seq_file)

        self.offsets_ppm = np.array(self.seq.dict_definitions['offsets_ppm'])
        self.n_offsets = self.offsets_ppm.size

        if 'num_meas' in self.seq.dict_definitions:
            self.n_measure = int(self.seq.dict_definitions['num_meas'])
        else:
            self.n_measure = self.n_offsets
            if 'run_m0_scan' in self.seq.dict_definitions:
                if 1 in self.seq.dict_definitions['run_m0_scan'] or 'True' in self.seq.dict_definitions['run_m0_scan']:
                    self.n_measure += 1

        self.m_init = params.m_vec.copy()
        self.m_out = np.zeros([self.m_init.shape[0], self.n_measure])

    def prep_rf_simulation(self, block):
        """
        Resamples the amplitude and phase of given rf event.
        :param block: rf event (block event)
        :return: amplitude, phase, duration and after_pulse_delay for given rf event
        """
        max_pulse_samples = self.params.options['max_pulse_samples']
        amp = np.abs(block.rf.signal)
        ph = np.angle(block.rf.signal)
        rf_length = amp.size
        dtp = 1e-6

        idx = np.argwhere(amp > 1E-6)
        amp = amp[idx]
        ph = ph[idx]
        delay_after_pulse = (rf_length - idx.size) * dtp
        n_unique = max(np.unique(amp).size, np.unique(ph).size)
        if n_unique == 1:
            amp_ = amp[0]
            ph_ = ph[0]
            dtp_ = dtp * amp.size
        elif n_unique > max_pulse_samples:
            sample_factor = int(np.ceil(amp.size / max_pulse_samples))
            amp_ = amp[::sample_factor]
            ph_ = ph[::sample_factor]
            dtp_ = dtp * sample_factor
        else:
            raise Exception('Case with 1 < unique samples < max_pulse_samples not implemented yet. Sorry :(')

        return amp_, ph_, dtp_, delay_after_pulse

    def run(self):
        """
        Creates BlochMcConnellSolver object and starts either the parallelized or the sequential simulation process.
        """
        self.bm_solver = BlochMcConnellSolver(params=self.params, n_offsets=self.n_offsets)
        if self.par_calc:
            self.run_parallel()
        else:
            self.run_sequential()

    def run_parallel(self):
        """
        Performs parallel simulation of all offsets.
        """

        if not self.params.options['reset_init_mag']:
            raise Exception("Parallel computation not possible for 'reset_init_mag = False'.\n"
                            "Please switch 'reset_init_mag' to 'True' or change to sequential computation.")

        # get dict with block events
        block_events = self.seq.dict_block_events

        # counter for even blocks related to an unsaturated M0 scan
        m0_event_count = 0

        # some preparations for cases with a leading unsaturated M0 scan
        if self.n_offsets != self.n_measure:
            if self.n_offsets == self.n_measure - 1:
                if self.verbose:
                    print(f"Number of measurements is exactly 1 larger than number of offsets. Continue with parallel "
                          f"computation assuming 1 leading unsaturated M0 scan.\nIf this is not correct, please restart "
                          f"the simulation with 'parc_calc = False' to switch to a sequential computation!")

                # set flag for unsaturated M0 scan
                self.run_m0_scan = True

                # create list with m0 block events
                m0_block_events = []

                # get event dict and number of events before the ADC event of the unsaturated M0 scan. Reading these
                # information directly from the seq-file is much faster than using the pypulseq Sequence.read() method.
                seq_file = open(self.seq_file, 'r')
                while True:
                    line = strip_line(seq_file)
                    if line == -1:
                        break

                    elif line == '[BLOCKS]':
                        line = strip_line(seq_file)

                        while line != '' and line != ' ' and line != '#':
                            block_event = np.fromstring(line, dtype=int, sep=' ')
                            m0_event_count += 1  # count number of events before 1st adc
                            m0_block_events.append(block_event)
                            if block_event[6]:
                                break
                            else:
                                line = strip_line(seq_file)
                    else:
                        pass
            else:
                raise Exception(f"Number of measurements and number of offsets differ by more than 1. Such a case is "
                                f"not suitable for parallelized computation. \nPlease switch to sequential "
                                f"computation by changing the 'par_calc' option from 'True' to 'False'.")

        # calculate number of event blocks EXCLUDING all unsaturated m0 events
        n_block_events = len(block_events) - m0_event_count

        # get number of blocks per offsets
        n_ = n_block_events / self.n_offsets
        if not n_.is_integer():
            raise Exception(f"Calculated number of block events per offset ({n_block_events}/{self.n_offsets} = {n_}) "
                            f"is not an integer. Aborting parallel computation.")
        n_: int = int(n_)

        # get dict with all block events of 2nd offset. This will be applied (with adjusted freq) to all offsets. The
        # 2nd offset (and not the 1st) is used because the 1st one is often a saturated M0 scan (e.g. at -300 ppm) with
        # a longer preceding recovery time. However, assuming that the block events of the 2nd offset can be applied to
        # all other offsets is a very strong assumption. Please be sure that you are aware of this when using par_calc.
        event_table_single_offset = {k: block_events[k] for k in list(block_events)[m0_event_count+n_:m0_event_count+2*n_]}

        # extract the offsets in rad from rf library
        events_freq = [self.seq.rf_library.data[k][4] for k in list(self.seq.rf_library.data)]
        events_phase = [self.seq.rf_library.data[k][5] for k in list(self.seq.rf_library.data)]

        # check if 0 ppm is in the events. Because all rf events with freq = 0 ppm have the same phase value of 0
        # (independent of the number of pulses per saturation train), only one single rf block event appears. For the
        # parallel computation, this event has to be duplicated and inserted into the event block dict until the
        # number of entries matches the number of entries at the other offsets.
        if 0.0 in events_freq:
            n_rf_per_offset = (len(events_freq) - 1) / (self.n_offsets - 1)
            if n_rf_per_offset.is_integer():
                n_rf_per_offset = int(n_rf_per_offset)
            else:
                raise Exception(
                    'Unexpected number of block events. The current scenario is probably not suitable for '
                    'the parallel computation in the current form')

            if n_rf_per_offset > 1:
                idx_zero = events_freq.index(0.0)
                events_freq[idx_zero:idx_zero] = [0.0] * (n_rf_per_offset - 1)
                events_phase[idx_zero:idx_zero] = [events_freq[idx_zero]] * (n_rf_per_offset - 1)

        else:
            n_rf_per_offset = len(events_freq) / self.n_offsets
            if n_rf_per_offset.is_integer():
                n_rf_per_offset = int(n_rf_per_offset)
            else:
                raise Exception(
                    'Unexpected number of block events. The current scenario is probably not suitable for '
                    'the parallel computation in the current form')

        # double check that the number of rf block events with a unique frequency matches the number of offsets
        offsets_hz = np.unique(events_freq)
        if len(offsets_hz) != len(self.offsets_ppm):
            raise Exception(
                f"Number of offsets from seq-file definitions ({len(self.offsets_ppm)}) don't match the "
                f"number of unique offsets ({len(offsets_hz)}) extracted from seq-file rf library.")

        # reshape phase events
        ph_offset = np.array(events_phase).reshape(self.n_offsets, n_rf_per_offset)

        # reshape magnetization vector
        M_ = np.repeat(self.m_init[np.newaxis, :, np.newaxis], self.n_offsets, axis=0)

        # handle m0 scan separately
        if self.run_m0_scan:
            m0 = M_.copy()
            for m0_event in m0_block_events:
                m0_block = self.seq.get_block(m0_event[0])
                if hasattr(m0_block, 'delay') and hasattr(m0_block.delay, 'delay'):
                    m0_delay = float(m0_block.delay.delay)
                    self.bm_solver.update_matrix(rf_amp=0.0,
                                                 rf_phase=np.zeros(self.n_offsets),
                                                 rf_freq=np.zeros(self.n_offsets))
                    m0 = self.bm_solver.solve_equation(mag=m0, dtp=m0_delay)

        # perform parallel computation
        rf_count = 0
        accum_phase = np.zeros(self.n_offsets)

        if self.params.options['verbose']:
            loop = tqdm(event_table_single_offset)
        else:
            loop = event_table_single_offset

        for x in loop:
            block = self.seq.get_block(x)

            if hasattr(block, 'adc'):
                m_out = np.swapaxes(np.squeeze(M_), 0, 1)
                if self.run_m0_scan:
                    m_out = np.concatenate((m0[0], m_out), axis=1)
                self.m_out = m_out

            elif hasattr(block, 'delay') and hasattr(block.delay, 'delay'):
                dtp_ = float(block.delay.delay)
                self.bm_solver.update_matrix(rf_amp=0.0,
                                             rf_phase=np.zeros(self.n_offsets),
                                             rf_freq=np.zeros(self.n_offsets))
                M_ = self.bm_solver.solve_equation(mag=M_, dtp=dtp_)

            elif hasattr(block, 'rf'):
                amp_, ph_, dtp_, delay_after_pulse = self.prep_rf_simulation(block)
                for i in range(amp_.size):
                    ph_i = ph_[i] + ph_offset[:, rf_count] - accum_phase
                    self.bm_solver.update_matrix(rf_amp=amp_[i],
                                                 rf_phase=ph_i,
                                                 rf_freq=np.array(offsets_hz))
                    M_ = self.bm_solver.solve_equation(mag=M_, dtp=dtp_)

                if delay_after_pulse > 0:
                    self.bm_solver.update_matrix(rf_amp=0.0,
                                                 rf_phase=np.zeros(self.n_offsets),
                                                 rf_freq=np.zeros(self.n_offsets))
                    M_ = self.bm_solver.solve_equation(mag=M_, dtp=delay_after_pulse)

                phase_degree = dtp_ * amp_.size * 360 * np.array(offsets_hz)
                phase_degree = np.mod(phase_degree,
                                      np.ones_like(phase_degree) * 360)  # this is x % 360 for an array
                accum_phase = accum_phase + (phase_degree / 180 * np.pi)
                rf_count += 1

            elif hasattr(block, 'gx') and hasattr(block, 'gy') and hasattr(block, 'gz'):
                dur_ = float(block.gx.rise_time + block.gx.flat_time + block.gx.fall_time)
                self.bm_solver.update_matrix(rf_amp=0.0,
                                             rf_phase=np.zeros(self.n_offsets),
                                             rf_freq=np.zeros(self.n_offsets))
                M_ = self.bm_solver.solve_equation(mag=M_, dtp=dur_)
                M_[:, 0:(len(self.params.cest_pools) + 1) * 2] = 0.0  # assume complete spoiling
            else:
                pass

    def run_sequential(self):
        """
        Performs sequential simulation of all offsets.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 0
        accum_phase = 0
        M_ = self.m_init[np.newaxis, :, np.newaxis]
        if self.params.options['verbose']:
            loop = tqdm(range(1, len(self.seq.dict_block_events)+1))
        else:
            loop = range(1, len(self.seq.dict_block_events)+1)
        for n_sample in loop:
            block = self.seq.get_block(n_sample)
            if hasattr(block, 'adc'):
                self.m_out[:, current_adc] = np.squeeze(M_)  # write current mag in output array
                accum_phase = 0
                current_adc += 1
                if current_adc <= self.n_offsets and self.params.options['reset_init_mag']:
                    M_ = self.m_init[np.newaxis, :, np.newaxis]

            elif hasattr(block, 'gx') and hasattr(block, 'gy') and hasattr(block, 'gz'):
                dur_ = float(block.gx.rise_time + block.gx.flat_time + block.gx.fall_time)
                self.bm_solver.update_matrix(0, 0, 0)
                M_ = self.bm_solver.solve_equation(mag=M_, dtp=dur_)
                for j in range((len(self.params.cest_pools) + 1) * 2):
                    M_[0, j, 0] = 0.0  # assume complete spoiling

            elif hasattr(block, 'rf'):
                amp_, ph_, dtp_, delay_after_pulse = self.prep_rf_simulation(block)
                for i in range(amp_.size):
                    self.bm_solver.update_matrix(rf_amp=amp_[i],
                                                 rf_phase=ph_[i] + block.rf.phase_offset - accum_phase,
                                                 rf_freq=block.rf.freq_offset)
                    M_ = self.bm_solver.solve_equation(mag=M_, dtp=dtp_)

                if delay_after_pulse > 0:
                    self.bm_solver.update_matrix(0, 0, 0)
                    M_ = self.bm_solver.solve_equation(mag=M_, dtp=delay_after_pulse)

                phase_degree = dtp_ * amp_.size * 360 * block.rf.freq_offset
                phase_degree %= 360
                accum_phase += phase_degree / 180 * np.pi

            elif hasattr(block, 'delay') and hasattr(block.delay, 'delay'):
                delay = float(block.delay.delay)
                self.bm_solver.update_matrix(0, 0, 0)
                M_ = self.bm_solver.solve_equation(mag=M_, dtp=delay)

            else:  # single gradient -> simulated as delay
                pass

    def get_zspec(self, return_abs: bool = True):
        """
        Calculates/extracts the water z-spectrum.
        :param return_abs: if True, returns np.abs() of z-magnetization (mz)
        :return: offsets in ppm (type: np.ndarray), z-spec (type: np.ndarray)
        """
        if self.run_m0_scan:
            m0 = self.m_out[self.params.mz_loc, 0]
            m_ = self.m_out[self.params.mz_loc, 1:]
            mz = m_/m0
        else:
            mz = self.m_out[self.params.mz_loc, :]

        if self.offsets_ppm.size != mz.size:
            self.offsets_ppm = np.arange(0, mz.size)

        if return_abs:
            mz = np.abs(mz)
        else:
            mz = np.array(mz)

        return self.offsets_ppm, mz
