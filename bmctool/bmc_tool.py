"""
bmc_tool.py
    Tool to solve the Bloch-McConnell (BMC) equations using a (parallelized) eigenwert ansatz.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import pypulseq as pp
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
    idx = np.argwhere(amp > 1e-6)

    try:
        rf_length = amp.size
        dtp = block.rf.t[1] - block.rf.t[0]
        delay_after_pulse = (rf_length - idx.size) * dtp
    except AttributeError:
        rf_length = amp.size
        dtp = 1e-6
        delay_after_pulse = (rf_length - idx.size) * dtp

    amp = amp[idx]
    ph = ph[idx]
    n_unique = max(np.unique(amp).size, np.unique(ph).size)

    # block pulse for seq-files >= 1.4.0
    if n_unique == 1 and amp.size ==2:
        amp_ = amp[0]
        ph_ = ph[0]
        dtp_ = dtp
    # block pulse for seq-files < 1.4.0
    elif n_unique == 1:
        amp_ = amp[0]
        ph_ = ph[0]
        dtp_ = dtp * amp.size
    # shaped pulse
    elif n_unique > max_pulse_samples:
        sample_factor = int(np.ceil(amp.size / max_pulse_samples))
        amp_ = amp[::sample_factor]
        ph_ = ph[::sample_factor]
        dtp_ = dtp * sample_factor
    else:
        amp_ = amp
        ph_ = ph
        dtp_ = dtp

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
        self.verbose = verbose
        self.run_m0_scan = None
        self.bm_solver = None

        self.seq = pp.Sequence()
        self.seq.read(seq_file)

        try:
            self.defs = self.seq.definitions
        except AttributeError:
            self.defs = self.seq.dict_definitions

        self.offsets_ppm = np.array(self.defs["offsets_ppm"])
        self.n_offsets = self.offsets_ppm.size

        if "num_meas" in self.defs:
            self.n_measure = int(self.defs["num_meas"])
        else:
            self.n_measure = self.n_offsets
            if "run_m0_scan" in self.defs:
                if 1 in self.defs["run_m0_scan"] or "True" in self.defs["run_m0_scan"]:
                    self.n_measure += 1

        self.m_init = params.m_vec.copy()
        self.m_out = np.zeros([self.m_init.shape[0], self.n_measure])

        self.bm_solver = BlochMcConnellSolver(params=self.params, n_offsets=self.n_offsets)

    def update_params(self, params: Params) -> None:
        """
        Update Params and BlochMcConnellSolver.
        """
        self.params = params
        self.bm_solver.update_params(params)

    def run(self) -> None:
        """
        Start simulation process.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 0
        accum_phase = 0
        mag = self.m_init[np.newaxis, :, np.newaxis]

        try:
            block_events = self.seq.block_events
        except AttributeError:
            block_events = self.seq.dict_block_events

        if self.verbose:
            loop_block_events = tqdm(range(1, len(block_events) + 1), desc="BMCTool simulation")
        else:
            loop_block_events = range(1, len(block_events) + 1)

        # code for pypulseq >= 1.4.0:
        try:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_4_0(block, current_adc, accum_phase, mag)
        except AttributeError:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_3_0(block, current_adc, accum_phase, mag)

    def run_1_4_0(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, np.ndarray]:
        # pseudo ADC event
        if block.adc is not None:
            # write current magnetization to output
            self.m_out[:, current_adc] = np.squeeze(mag)
            accum_phase = 0
            current_adc += 1
            # reset mag if this wasn't the last ADC event
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

        # spoiler gradient in z-direction
        elif block.gz is not None:
            dur_ = block.block_duration
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
            for j in range((len(self.params.cest_pools) + 1) * 2):
                mag[0, j, 0] = 0.0  # assume complete spoiling

        # delay or gradient(s) in x and/or y-direction
        elif hasattr(block, "block_duration") and block.block_duration != "0":
            delay = block.block_duration
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

        # this should not happen
        else:
            raise Exception("Unknown case")

        return current_adc, accum_phase, mag

    def run_1_3_0(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, np.ndarray]:
        # pseudo ADC event
        if hasattr(block, "adc"):
            # write current magnetization to output
            self.m_out[:, current_adc] = np.squeeze(mag)
            accum_phase = 0
            current_adc += 1
            # reset mag if this wasn't the last ADC event
            if current_adc <= self.n_offsets and self.params.options["reset_init_mag"]:
                mag = self.m_init[np.newaxis, :, np.newaxis]

        # RF pulse
        elif hasattr(block, "rf"):
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

        # spoiler gradient in z-direction
        elif hasattr(block, "gz"):
            dur_ = float(block.gz.rise_time + block.gz.flat_time + block.gz.fall_time)
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
            for j in range((len(self.params.cest_pools) + 1) * 2):
                mag[0, j, 0] = 0.0  # assume complete spoiling

        # gradient in x and/or y-direction (handled as delay)
        elif hasattr(block, "gx") or hasattr(block, "gy"):
            if hasattr(block, "gx"):
                delay = float(block.gx.rise_time + block.gx.flat_time + block.gx.fall_time)
            elif hasattr(block, "gy"):
                delay = float(block.gy.rise_time + block.gy.flat_time + block.gy.fall_time)
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

        # delay
        elif hasattr(block, "delay") and hasattr(block.delay, "delay"):
            delay = float(block.delay.delay)
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

        # this should not happen
        else:
            raise Exception("Unknown case")

        return current_adc, accum_phase, mag

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
