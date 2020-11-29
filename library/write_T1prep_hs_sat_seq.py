"""
Script to output a seq file for a T1 prep with an adiabatic hyperbolic secant preparation pulse.
"""

import os
import numpy as np
from datetime import date
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.opts import Opts
from sim.utils.seq.conversion import convert_seq_12_to_pseudo_13
from sim.utils.seq.make_hypsec_half_passage_rf import make_hypsec_half_passage_rf

# ========
# SETTINGS
# ========

# seq-file name
seq_filename = 'T1prep_HSsat.seq'

# plot preparation block?
plot_sequence = True

# convert seq-file to a pseudo version 1.3 file?
convert_to_1_3 = True

# inversion times (number of inversion times defines the number of measurements)
TI_times = np.array([10, 6, 5, 4, 3, 2.5, 2, 1.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

# settings of adiabatic hyperbolic secant (hs) preparation pulse(s)
b1 = 20  # HS pulse amplitude [µT]
t_p_hs = 8e-3  # HS pulse duration [s]
n_hs_pulses = 3  # number of (HS pulse + spoiler) blocks

# settings for simulation tools
run_m0_scan = False  # for compatibility

# scanner settings
b0 = 3  # B0 [T]
sys = Opts(max_grad=40, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
           rf_ringdown_time=30e-6, rf_dead_time=100e-6, rf_raster_time=1e-6)

# ===========
# PREPARATION
# ===========

# spoilers
spoil_amp0 = 0.8 * sys.max_grad  # Hz/m
spoil_amp1 = -0.7 * sys.max_grad  # Hz/m
spoil_amp2 = 0.6 * sys.max_grad  # Hz/m

rise_time = 1.0e-3  # spoiler rise time in seconds
spoil_dur = 5.5e-3  # complete spoiler duration in seconds

gx_spoil0, gy_spoil0, gz_spoil0 = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp0, duration=spoil_dur,
                                                  rise_time=rise_time) for c in ['x', 'y', 'z']]
gx_spoil1, gy_spoil1, gz_spoil1 = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp1, duration=spoil_dur,
                                                  rise_time=rise_time) for c in ['x', 'y', 'z']]
gx_spoil2, gy_spoil2, gz_spoil2 = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp2, duration=spoil_dur,
                                                  rise_time=rise_time) for c in ['x', 'y', 'z']]

# RF pulses
hs_pulse = make_hypsec_half_passage_rf(amp=b1, system=sys)

# ADC events
pseudo_adc = make_adc(num_samples=1, duration=1e-3)  # (not played out; just used to split measurements)

# DELAYS
initial_delay = make_delay(1e-3)
post_spoil_delay = make_delay(50e-6)

# Sequence object
seq = Sequence()

# ===
# RUN
# ===

for t_prep in TI_times:
    # add initial delay
    seq.add_block(initial_delay)
    # add preparation (adiabatic excitation + spoiler) block
    for i in range(n_hs_pulses):
        seq.add_block(hs_pulse)
        if i % 3 == 0:
            seq.add_block(gx_spoil0, gy_spoil1, gz_spoil2)
        elif i % 2 == 0:
            seq.add_block(gx_spoil2, gy_spoil0, gz_spoil1)
        else:
            seq.add_block(gx_spoil1, gy_spoil2, gz_spoil0)
    # add variable inversion time delay
    seq.add_block(make_delay(t_prep))
    # add pseudo adc
    seq.add_block(pseudo_adc)

# add definitions to seq-file
seq.set_definition('offsets_ppm', np.zeros(TI_times.shape))  # required
seq.set_definition('run_m0_scan', str(run_m0_scan))  # required
seq.set_definition('TI_times', TI_times)
seq.set_definition('seq_name', seq_filename)
seq.set_definition('date', date.today().strftime('%Y-%m-%d'))

# plot the sequence
if plot_sequence:
    seq.plot()
print(seq.shape_library)

# write seq-file
seq_file_path = os.path.join('seq-library', seq_filename)
seq.write(seq_file_path)

# convert to pseudo version 1.3
if convert_to_1_3:
    convert_seq_12_to_pseudo_13(seq_file_path)
