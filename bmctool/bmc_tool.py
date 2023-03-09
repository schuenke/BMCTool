"""
bmc_tool.py
    Tool to solve the Bloch-McConnell (BMC) equations using a (parallelized) eigenwert ansatz.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
from pypulseq import Sequence
from pypulseq.Sequence.read_seq import __strip_line as strip_line
from tqdm import tqdm

from bmctool.bmc_solver import BlochMcConnellSolver
from bmctool.params import Params


def prep_rf_simulation(block: SimpleNamespace, max_pulse_samples: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    prep_rf_simulation Resamples the amplitude and phase of given rf event.

    Parameters
    ----------
    block : SimpleNamespace
        PyPulseq block event
    max_pulse_samples : int
        Maximum number of samples for the rf pulse

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, float]
        Tuple of resampled amplitude, phase, time step and delay after pulse

    Raises
    ------
    Exception
        If number of unique samples is larger than 1 but smaller than max_pulse_samples (not implemented yet)
    """
    amp = np.abs(block.rf.signal)
    ph = np.angle(block.rf.signal)
    rf_length = block.rf.shape_dur
    dtp = rf_length / max(amp.size, ph.size)

    idx = np.argwhere(amp > 1e-6)
    amp = amp[idx]
    ph = ph[idx]
    delay_after_pulse = block.block_duration - block.rf.shape_dur
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
        raise Exception("Case with 1 < unique samples < max_pulse_samples not implemented yet. Sorry :(")

    return amp_, ph_, dtp_, delay_after_pulse


class BMCTool:
    """
    Definition of the BMCTool class.
    """

    def __init__(self, params: Params, seq_file: Union[str, Path], verbose: bool = True, **kwargs) -> None:
        """
        __init__ Initialize BMCTool object.

        Parameters
        ----------
        params : Params
            Params object containing all simulation parameters
        seq_file : Union[str, Path]
            Path to the seq-file
        verbose : bool, optional
            Flag to activate detailed outpus, by default True
        """
        self.params = params
        self.seq_file = seq_file
        self.par_calc = params.options["par_calc"]
        self.verbose = verbose
        self.run_m0_scan = None
        self.bm_solver = None

        self.seq = Sequence()
        self.seq.read(seq_file)

        self.offsets_ppm = np.array(self.seq.definitions["offsets_ppm"])
        self.n_offsets = self.offsets_ppm.size

        if "num_meas" in self.seq.definitions:
            self.n_measure = int(self.seq.definitions["num_meas"])
        else:
            self.n_measure = self.n_offsets
            if "run_m0_scan" in self.seq.definitions:
                if 1 in self.seq.definitions["run_m0_scan"] or "True" in self.seq.definitions["run_m0_scan"]:
                    self.n_measure += 1

        self.m_init = params.m_vec.copy()
        self.m_out = np.zeros([self.m_init.shape[0], self.n_measure])

        self.bm_solver = BlochMcConnellSolver(params=self.params, n_offsets=self.n_offsets)

    def update_params(
        self,
        params: Params,
    ) -> None:
        """
        Update Params and BlochMcConnellSolver.
        """
        self.params = params
        self.bm_solver.update_params(params)

    def run(self) -> None:
        """
        Start either parallelized or the sequential simulation process.
        """
        if self.par_calc:
            if self.verbose:
                print("\n-> Starting parallel simulation...")
            self.run_parallel()
        else:
            if self.verbose:
                print("\n-> Starting sequential simulation...")
            self.run_sequential()

    def run_parallel(self) -> None:
        """
        Perform simulation of all block events for all offsets in parallel.
        """

        if not self.params.options["reset_init_mag"]:
            raise Exception(
                "Parallel computation not possible for 'reset_init_mag = False'.\n"
                "Please switch 'reset_init_mag' to 'True' or change to sequential computation."
            )

        # get dict with block events
        block_events = self.seq.block_events

        # counter for event blocks related to an unsaturated M0 scan
        m0_event_count = 0

        # some preparations for cases with a leading unsaturated M0 scan
        if self.n_offsets != self.n_measure:
            if self.n_offsets == self.n_measure - 1:
                if self.verbose:
                    print(
                        "Number of measurements is exactly 1 larger than number of offsets. Continue with parallel computation assuming 1 leading unsaturated M0 scan.\nIf this is not correct, please restart the simulation with 'parc_calc = False' to switch to a sequential computation!"
                    )

                # set flag for unsaturated M0 scan
                self.run_m0_scan = True

                # create list with m0 block events
                m0_block_events = []

                # get event dict and number of events before the ADC event of the unsaturated M0 scan. Reading these
                # information directly from the seq-file is much faster than using the pypulseq Sequence.read() method.
                seq_file = open(self.seq_file, "r")
                while True:
                    line = strip_line(seq_file)
                    if line == -1:
                        break

                    elif line == "[BLOCKS]":
                        line = strip_line(seq_file)

                        while line != "" and line != " " and line != "#":
                            block_event = np.fromstring(line, dtype=int, sep=" ")
                            m0_event_count += 1  # count number of events before 1st adc
                            m0_block_events.append(block_event)
                            if block_event[6]:
                                break
                            else:
                                line = strip_line(seq_file)
                    else:
                        pass
            else:
                raise Exception(
                    "Number of measurements and number of offsets differ by more than 1. Such a case is not suitable for parallelized computation. \nPlease switch to sequential computation by changing the 'par_calc' option from 'True' to 'False'."
                )

        # calculate number of event blocks EXCLUDING all unsaturated m0 events
        n_block_events = len(block_events) - m0_event_count

        # get number of blocks per offsets
        n_blocks_per_offset = n_block_events / self.n_offsets
        if not n_blocks_per_offset.is_integer():
            raise Exception(
                f"Calculated number of block events per offset ({n_block_events}/{self.n_offsets} = {n_blocks_per_offset}) "
                f"is not an integer. Aborting parallel computation."
            )
        n_blocks_per_offset: int = int(n_blocks_per_offset)

        # get dict with all block events of 2nd offset. This will be applied (with adjusted freq) to all offsets. The
        # 2nd offset (and not the 1st) is used because the 1st one is often a saturated M0 scan (e.g. at -300 ppm) with
        # a longer preceding recovery time. However, assuming that the block events of the 2nd offset can be applied to
        # all other offsets is a very strong assumption. Please be sure that you are aware of this when using par_calc.
        event_table_single_offset = {
            k: block_events[k]
            for k in list(block_events)[m0_event_count + n_blocks_per_offset : m0_event_count + 2 * n_blocks_per_offset]
        }

        # extract the offsets in rad from rf library
        events_freq = [self.seq.rf_library.data[k][5] for k in list(self.seq.rf_library.data)]
        events_phase = [self.seq.rf_library.data[k][6] for k in list(self.seq.rf_library.data)]

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
                    "Unexpected number of block events. The current scenario is probably not suitable for the parallel computation in the current form"
                )

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
                    "Unexpected number of block events. The current scenario is probably not suitable for the parallel computation in the current form"
                )

        # double check that the number of rf block events with a unique frequency matches the number of offsets
        offsets_hz = np.unique(events_freq)
        if len(offsets_hz) != len(self.offsets_ppm):
            raise Exception(
                f"Number of offsets from seq-file definitions ({len(self.offsets_ppm)}) don't match the number of unique offsets ({len(offsets_hz)}) extracted from seq-file rf library."
            )

        # reshape phase events
        ph_offset = np.array(events_phase).reshape(self.n_offsets, n_rf_per_offset)

        # reshape magnetization vector
        mag = np.repeat(self.m_init[np.newaxis, :, np.newaxis], self.n_offsets, axis=0)

        # handle m0 scan separately
        if self.run_m0_scan:
            m_0 = mag.copy()
            for m0_event in m0_block_events:
                m0_block = self.seq.get_block(m0_event[0])
                if hasattr(m0_block, "delay") and hasattr(m0_block.delay, "delay"):
                    m0_delay = float(m0_block.delay.delay)
                    self.bm_solver.update_matrix(
                        rf_amp=0.0, rf_phase=np.zeros(self.n_offsets), rf_freq=np.zeros(self.n_offsets)
                    )
                    m_0 = self.bm_solver.solve_equation(mag=m_0, dtp=m0_delay)

        # perform parallel computation
        rf_count = 0
        accum_phase = np.zeros(self.n_offsets)

        if self.verbose:
            loop_block_events = tqdm(event_table_single_offset)
        else:
            loop_block_events = event_table_single_offset

        for block_event in loop_block_events:
            block = self.seq.get_block(block_event)

            # pseudo ADC event
            if block.adc is not None:
                m_out = np.swapaxes(np.squeeze(mag), 0, 1)
                if self.run_m0_scan:
                    m_out = np.concatenate((m_0[0], m_out), axis=1)
                self.m_out = m_out

            # RF pulse
            elif hasattr(block, "rf") and block.rf is not None:
                amp_, ph_, dtp_, delay_after_pulse = prep_rf_simulation(block, self.params.options["max_pulse_samples"])
                for i in range(amp_.size):
                    ph_i = -ph_[i] + ph_offset[:, rf_count] - accum_phase
                    self.bm_solver.update_matrix(rf_amp=amp_[i], rf_phase=ph_i, rf_freq=np.array(offsets_hz))
                    mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)

                if delay_after_pulse > 0:
                    self.bm_solver.update_matrix(
                        rf_amp=0.0, rf_phase=np.zeros(self.n_offsets), rf_freq=np.zeros(self.n_offsets)
                    )
                    mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

                phase_degree = dtp_ * amp_.size * 360 * np.array(offsets_hz)
                phase_degree = np.mod(phase_degree, np.ones_like(phase_degree) * 360)  # this is x % 360 for an array
                accum_phase = accum_phase + (phase_degree / 180 * np.pi)
                rf_count += 1

            # spoiler gradients in x,y,z
            elif all(b is not None for b in [block.gx, block.gy, block.gz]):
                dur_ = block.block_duration
                self.bm_solver.update_matrix(
                    rf_amp=0.0, rf_phase=np.zeros(self.n_offsets), rf_freq=np.zeros(self.n_offsets)
                )
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
                mag[:, 0 : (len(self.params.cest_pools) + 1) * 2] = 0.0  # assume complete spoiling

            # spoiler gradient in z only
            elif block.gz is not None:
                dur_ = block.block_duration
                self.bm_solver.update_matrix(
                    rf_amp=0.0, rf_phase=np.zeros(self.n_offsets), rf_freq=np.zeros(self.n_offsets)
                )
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
                mag[:, 0 : (len(self.params.cest_pools) + 1) * 2] = 0.0  # assume complete spoiling

            # delay or gradient(s) in x and/or y --> handle as delay
            elif hasattr(block, "block_duration") and block.block_duration != "0":
                delay = block.block_duration
                self.bm_solver.update_matrix(
                    rf_amp=0.0, rf_phase=np.zeros(self.n_offsets), rf_freq=np.zeros(self.n_offsets)
                )
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

            # this should not happen
            else:
                raise Exception("Unknown case")

    def run_sequential(self) -> None:
        """
        Performs simulation of all block events for all offsets sequentially.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 0
        accum_phase = 0
        mag = self.m_init[np.newaxis, :, np.newaxis]
        if self.verbose:
            loop_block_events = tqdm(range(1, len(self.seq.block_events) + 1), desc="BMCTool simulation")
        else:
            loop_block_events = range(1, len(self.seq.block_events) + 1)

        for block_event in loop_block_events:
            block = self.seq.get_block(block_event)

            # pseudo ADC event
            if block.adc is not None:
                self.m_out[:, current_adc] = np.squeeze(mag)  # write current mag in output array
                accum_phase = 0
                current_adc += 1
                if current_adc <= self.n_offsets and self.params.options["reset_init_mag"]:
                    mag = self.m_init[np.newaxis, :, np.newaxis]

            # RF pulse
            elif block.rf is not None:
                amp_, ph_, dtp_, delay_after_pulse = prep_rf_simulation(block, self.params.options["max_pulse_samples"])
                for i in range(amp_.size):
                    self.bm_solver.update_matrix(
                        rf_amp=amp_[i],
                        rf_phase=-ph_[i] + block.rf.phase_offset - accum_phase,
                        rf_freq=block.rf.freq_offset,
                    )
                    mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)

                if delay_after_pulse > 0:
                    self.bm_solver.update_matrix(0, 0, 0)
                    mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

                phase_degree = dtp_ * amp_.size * 360 * block.rf.freq_offset
                phase_degree %= 360
                accum_phase += phase_degree / 180 * np.pi

            # spoiler gradients in x,y,z
            elif all(b is not None for b in [block.gx, block.gy, block.gz]):
                dur_ = block.block_duration
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
                for j in range((len(self.params.cest_pools) + 1) * 2):
                    mag[0, j, 0] = 0.0  # assume complete spoiling

            # spoiler gradient in z only
            elif block.gz is not None:
                dur_ = block.block_duration
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
                for j in range((len(self.params.cest_pools) + 1) * 2):
                    mag[0, j, 0] = 0.0  # assume complete spoiling

            # delay or gradient(s) in x and/or y --> handle as delay
            elif hasattr(block, "block_duration") and block.block_duration != "0":
                delay = block.block_duration
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

            # this should not happen
            else:
                raise Exception("Unknown case")

    def get_zspec(self, return_abs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        get_zspec Calculate/extract the Z-spectrum.

        Parameters
        ----------
        return_abs : bool, optional
            flag to activate the return of absolute values, by default True

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of offsets and Z-spectrum
        """
        if self.run_m0_scan:
            m_0 = self.m_out[self.params.mz_loc, 0]
            m_ = self.m_out[self.params.mz_loc, 1:]
            m_z = m_ / m_0
        else:
            m_z = self.m_out[self.params.mz_loc, :]

        if self.offsets_ppm.size != m_z.size:
            self.offsets_ppm = np.arange(0, m_z.size)

        if return_abs:
            m_z = np.abs(m_z)
        else:
            m_z = np.array(m_z)

        return self.offsets_ppm, m_z
