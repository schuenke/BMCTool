"""
Script to output a seq file for a WASABI protocol for simultaneous mapping of B0 and B1 according to:
Schuenke et al. Simultaneous mapping of water shift and B1 (WASABI)-Application to field-Inhomogeneity correction of
CEST MRI data. Magnetic Resonance in Medicine, 77(2), 571–580. https://doi.org/10.1002/mrm.26133
parameter settings:
     pulse shape = block
     B1 = 3.75 uT (@ 3T; in general 1.25 uT multiplied by field strength)
     n = 1
     t_p = 5 ms (@ 3T; in general 15 ms devided by field strength)
     T_rec = 2/12 s (saturated/M0)
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

# ========
# SETTINGS
# ========

# seq-file name
seq_filename = 'WASABI.seq'

# plot preparation block?
plot_sequence = True

# convert seq-file to a pseudo version 1.3 file?
convert_to_1_3 = True

# offset settings
offset_range = 2  # [ppm]
num_offsets = 31  # number of measurements (not including M0)
offsets_ppm = np.linspace(-offset_range, offset_range, num_offsets)

run_m0_scan = True  # if you want an M0 scan at the beginning
m0_offset = False  # ppm

# sequence settings
t_rec = 2  # recovery time between scans [s]
m0_t_rec = 12  # recovery time before m0 scan [s]
b1 = 3.75  # mean sat pulse amp [uT]
t_p = 0.005  # sat pulse duration [s]

# scanner settings
b0 = 3  # B0 [T]
sys = Opts(max_grad=40, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
           rf_ringdown_time=30e-6, rf_dead_time=100e-6, rf_raster_time=1e-6)

# ===========
# PREPARATION
# ===========

# spoiler
spoil_amp = 0.8 * sys.max_grad  # Hz/m
rise_time = 1.0e-3  # spoiler rise time in seconds
spoil_dur = 5.5e-3  # complete spoiler duration in seconds

gx_spoil, gy_spoil, gz_spoil = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp, duration=spoil_dur,
                                               rise_time=rise_time) for c in ['x', 'y', 'z']]

# RF pulses
flip_angle_wasabi = b1 * sys.gamma * 1e-6 * 2 * np.pi * t_p
rf_wasabi, _ = make_block_pulse(flip_angle=flip_angle_wasabi, duration=t_p, system=sys)

# ADC events
pseudo_adc = make_adc(num_samples=1, duration=1e-3)  # (not played out; just used to split measurements)

# DELAYS
post_spoil_delay = make_delay(50e-6)
trec_delay = make_delay(t_rec)
m0_delay = make_delay(m0_t_rec)

# Sequence object
seq = Sequence()

# ===
# RUN
# ===

# run m0 scan with or without the defined offset
if run_m0_scan:
    seq.add_block(m0_delay)
    seq.add_block(pseudo_adc)

offsets = offsets_ppm * sys.gamma * 1e-6 * b0  # convert from ppm to rad

for offset in offsets:
    # add magnetization recover delay
    if abs(offset) > 295:
        seq.add_block(m0_delay)
    else:
        seq.add_block(trec_delay)

    # update frequency offset of WASABI pulse and add the rf block
    rf_wasabi.freq_offset = offset
    seq.add_block(rf_wasabi)

    # add spoiler and post-spoiling delay
    seq.add_block(gx_spoil, gy_spoil, gz_spoil)
    seq.add_block(post_spoil_delay)

    # add pseudo adc
    seq.add_block(pseudo_adc)

# add definitions to seq-file
seq.set_definition('offsets_ppm', offsets_ppm)  # required
seq.set_definition('run_m0_scan', str(run_m0_scan))  # required
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
