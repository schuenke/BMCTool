from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from tqdm import tqdm

from bmctool.parameters import Parameters
from bmctool.simulation._BlochMcConnellSolver import BlochMcConnellSolver


class BMCSim:
    """Class for Bloch-McConnell simulations using PyPulseq sequences."""

    def __init__(
        self,
        params: Parameters,
        seq: str | Path | pp.Sequence,
        verbose: bool = True,
    ) -> None:
        """Initialize BMCSim object.

        Parameters
        ----------
        params
            Parameters object containing all simulation parameters
        seq
            Path to the pulseq seq-file or PyPulseq sequence object
        verbose, optional
            Flag to activate detailed outputs, by default True
        """
        self.params = params
        self.verbose = verbose

        # load sequence
        if isinstance(seq, pp.Sequence):
            self.seq = seq
        else:
            self.seq = pp.Sequence()
            self.seq.read(seq)

        # get offsets from pypulseq definitions
        self.defs = self.seq.definitions
        self.offsets_ppm = np.array(self.defs['offsets_ppm'])
        self.n_measure = self.offsets_ppm.size

        # extract initial magnetization vector and create output array
        self.m_init = params.m_vec.copy()
        self.m_out = np.zeros([self.m_init.shape[0], self.n_measure])

        # initialize solver
        self.bm_solver = BlochMcConnellSolver(params=self.params, n_offsets=self.n_measure)

    @staticmethod
    def prep_rf_simulation(
        block: SimpleNamespace,
        max_pulse_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Resample amplitude and phase of an RF event within given block.

        Parameters
        ----------
        block
            PyPulseq block object containing the RF event
        max_pulse_samples
            Maximum number of samples for the resampled rf pulse

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, float]
            Tuple of resampled amplitude, phase, time step and delay after pulse
        """
        # get amplitude and phase of RF pulse
        amp = np.abs(block.rf.signal)
        ph = np.angle(block.rf.signal)

        # find all non-zero sample points
        idx = np.argwhere(amp > 1e-6)

        # calculate time step and delay after pulse
        rf_length = amp.size
        dtp = block.rf.t[1] - block.rf.t[0]
        delay_after_pulse = (rf_length - idx.size) * dtp

        # remove all zero samples
        amp = amp[idx]
        ph = ph[idx]

        # get maximum number of unique samples in amplitude and/or phase
        n_unique = max(np.unique(amp).size, np.unique(ph).size)

        # block pulse for seq-files >= 1.4.0
        if n_unique == 1 and amp.size == 2:
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

    def update_params(self, params: Parameters) -> None:
        """Update Params and BlochMcConnellSolver."""
        self.params = params
        self.bm_solver.update_params(params)

    def run(self) -> None:
        """Start simulation process."""
        current_adc = 0
        accum_phase = 0.0

        # create initial magnezitation array with correct shape
        mag = self.m_init[np.newaxis, :, np.newaxis]

        # get all block events from pypulseq sequence
        block_events = self.seq.block_events

        # create loop with or w/o tqdm status bar depending on verbose settings
        if self.verbose:
            loop_block_events = tqdm(range(1, len(block_events) + 1), desc='BMCTool simulation')
        else:
            loop_block_events = range(1, len(block_events) + 1)

        # run simulation for all blocks
        for block_event in loop_block_events:
            block = self.seq.get_block(block_event)
            current_adc, accum_phase, mag = self._simulate_block(block, current_adc, accum_phase, mag)

    def _simulate_block(
        self,
        block: SimpleNamespace,
        current_adc: int,
        accum_phase: float,
        mag: np.ndarray,
    ) -> tuple[int, float, np.ndarray]:
        """Run BMC simulation for given block containing different events.

        The BMCTool distinguishes between the following cases:
        1) block contains an ADC: proceeding to next offset, all other events are neglected.
        2) block contains an RF, but no ADC: simulate the RF pulse, all other events are neglected.
        3) block contains a z-gradient, but no ADC or RF: assume spoiling, all other events are neglected.
        4) block contains no ADC, RF, z-grad, but x/y-grad or delay: simulate delay

        Parameters
        ----------
        block
            PyPulseq block object containing different events
        current_adc
            ADC counter
        accum_phase
            accumulated phase from previous RF pulses
        mag
            magnetization vector from previous block

        Returns
        -------
        Tuple[int, float, np.ndarray]
            Tuple of ADC counter, accumulated phase and magnetization vector

        Raises
        ------
        ValueError
            If the current block event cannot be handled by BMCTool
        """
        # Pseudo ADC event
        if block.adc is not None:
            current_adc, accum_phase, mag = self._handle_adc_event(current_adc, accum_phase, mag)

        # RF pulse
        elif block.rf is not None:
            current_adc, accum_phase, mag = self._handle_rf_pulse(block, current_adc, accum_phase, mag)

        # Spoiler gradient in z-direction
        elif block.gz is not None:
            mag = self._handle_spoiler_gradient(block, mag)

        # Delay or gradient(s) in x and/or y-direction
        elif hasattr(block, 'block_duration') and block.block_duration != '0':
            mag = self._handle_delay_or_gradient(block, mag)

        # This should not happen
        else:
            raise ValueError('The current block event cannot be handled by BMCTool. Please check you sequence.')

        return current_adc, accum_phase, mag

    def _handle_adc_event(self, current_adc: int, accum_phase: float, mag: np.ndarray) -> tuple[int, float, np.ndarray]:
        """Handle ADC event: write current mag to output, reset phase and increase ADC counter."""
        # write current magnetization to output
        self.m_out[:, current_adc] = np.squeeze(mag)

        # reset phase and increase ADC counter
        accum_phase = 0.0
        current_adc += 1

        # reset magnetization if reset_init_mag is True
        if self.params.options.reset_init_mag:
            mag = self.m_init[np.newaxis, :, np.newaxis]
        return current_adc, accum_phase, mag

    def _handle_rf_pulse(
        self,
        block: SimpleNamespace,
        current_adc: int,
        accum_phase: float,
        mag: np.ndarray,
    ) -> tuple[int, float, np.ndarray]:
        """Handle RF pulse: simulate all steps of RF pulse and update phase."""
        # resample amplitude and phase of RF pulse according to max_pulse_samples
        amp_, ph_, dtp_, delay_after_pulse = self.prep_rf_simulation(block, self.params.options.max_pulse_samples)

        # simulate all steps of RF pulse subsequently
        for i in range(amp_.size):
            self.bm_solver.update_matrix(
                rf_amp=amp_[i],
                rf_phase=-ph_[i] + block.rf.phase_offset - accum_phase,
                rf_freq=block.rf.freq_offset,
            )
            mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)

        # simulate a potential delay after the RF pulse
        if delay_after_pulse > 0:
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

        # update accumulated phase
        phase_degree = dtp_ * amp_.size * 360 * block.rf.freq_offset
        phase_degree %= 360
        accum_phase += phase_degree / 180 * np.pi

        return current_adc, accum_phase, mag

    def _handle_spoiler_gradient(self, block: SimpleNamespace, mag: np.ndarray) -> np.ndarray:
        """Handle spoiler gradient: assume complete spoiling."""
        _dur = block.block_duration
        self.bm_solver.update_matrix(0, 0, 0)
        mag = self.bm_solver.solve_equation(mag=mag, dtp=_dur)

        # set x and y components of the water pool and all cest pools to zero
        mag[0, : ((len(self.params.cest_pools) + 1) * 2), 0] = 0.0

        return mag

    def _handle_delay_or_gradient(self, block: SimpleNamespace, mag: np.ndarray) -> np.ndarray:
        """Handle delay or gradient(s): simulate delay."""
        _dur = block.block_duration
        self.bm_solver.update_matrix(0, 0, 0)
        mag = self.bm_solver.solve_equation(mag=mag, dtp=_dur)

        return mag

    def get_zspec(self, return_abs: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Calculate/extract the Z-spectrum.

        Parameters
        ----------
        return_abs, optional
            flag to activate the return of absolute values, by default True

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of offsets and Z-spectrum
        """
        m_z = self.m_out[self.params.mz_loc, :]

        if self.offsets_ppm.size != m_z.size:
            self.offsets_ppm = np.arange(0, m_z.size)

        m_z = np.abs(m_z) if return_abs else np.array(m_z)

        return self.offsets_ppm, m_z
