"""
Script to output a seq file for a WASABI protocol for simultaneous mapping of B0 and B1 according to:
Schuenke et al. Simultaneous mapping of water shift and B1 (WASABI)-Application to field-Inhomogeneity correction of
CEST MRI data. Magnetic Resonance in Medicine, 77(2), 571–580. https://doi.org/10.1002/mrm.26133
parameter settings:
     pulse shape = block
     B1 = 3.70 uT
     n = 1
     t_p = 5 ms
     T_rec = 3/12 s (saturated/M0)
"""

import numpy as np
from pypulseq import Opts
from pypulseq import Sequence
from pypulseq import make_adc
from pypulseq import make_block_pulse
from pypulseq import make_delay
from pypulseq import make_trapezoid

from src.bmctool.utils.seq.write import write_seq

# general settings
seqid = 'WASABI'
author = 'Patrick Schuenke'
plot_sequence = True  # plot preparation block?

# sequence definitions (everything in seq_defs will be written to definitions of the .seq-file)
seq_defs: dict = {}
seq_defs['b1cwpe'] = 3.7  # B1 amplitude [µT]
seq_defs['b0'] = 3  # B0 [T]
seq_defs['n_pulses'] = 1  # number of pulses  #
seq_defs['tp'] = 5e-3  # pulse duration [s]
seq_defs['trec'] = 3  # recovery time [s]
seq_defs['trec_m0'] = 12  # recovery time before M0 [s]
seq_defs['m0_offset'] = -300  # m0 offset [ppm]
seq_defs['offsets_ppm'] = np.append(seq_defs['m0_offset'], np.linspace(-2, 2, 31))  # offset vector [ppm]

seq_defs['num_meas'] = seq_defs['offsets_ppm'].size  # number of repetition
seq_defs['tsat'] = seq_defs['tp']  # saturation time [s]
seq_defs['seq_id_string'] = seqid  # unique seq id

seq_filename = seq_defs['seq_id_string'] + '.seq'

# scanner limits
sys = Opts(
    max_grad=40,
    grad_unit='mT/m',
    max_slew=130,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    rf_raster_time=1e-6,
)

gamma_hz = 42.5764

# ===========
# PREPARATION
# ===========

# spoiler
spoil_amp = 0.8 * sys.max_grad  # Hz/m
rise_time = 1.0e-3  # spoiler rise time in seconds
spoil_dur = 6.5e-3  # complete spoiler duration in seconds

gx_spoil, gy_spoil, gz_spoil = (
    make_trapezoid(
        channel=c,
        system=sys,
        amplitude=spoil_amp,
        duration=spoil_dur,
        rise_time=rise_time,
    )
    for c in ['x', 'y', 'z']
)

# RF pulses
flip_angle_sat = seq_defs['b1cwpe'] * gamma_hz * 2 * np.pi * seq_defs['tp']
rf_pulse = make_block_pulse(flip_angle=flip_angle_sat, duration=seq_defs['tp'], system=sys)

# ADC events
pseudo_adc = make_adc(num_samples=1, duration=1e-3)  # (not played out; just used to split measurements)

# DELAYS
trec_delay = make_delay(seq_defs['trec'])
m0_delay = make_delay(seq_defs['trec_m0'])

# Sequence object
seq = Sequence()

# ===
# RUN
# ===

offsets_hz = seq_defs['offsets_ppm'] * gamma_hz * seq_defs['b0']  # convert from ppm to Hz

for m, offset in enumerate(offsets_hz):
    # print progress/offset
    print(f' {m + 1} / {len(offsets_hz)} : offset {offset}')

    # add delay
    if offset == seq_defs['m0_offset'] * gamma_hz * seq_defs['b0']:
        if seq_defs['trec_m0'] > 0:
            seq.add_block(m0_delay)
    else:
        if seq_defs['trec'] > 0:
            seq.add_block(trec_delay)

    # set wasabi pulse
    rf_pulse.freq_offset = offset
    seq.add_block(rf_pulse)

    seq.add_block(gx_spoil, gy_spoil, gz_spoil)
    seq.add_block(pseudo_adc)

write_seq(
    seq=seq,
    seq_defs=seq_defs,
    filename=seqid + '.seq',
    author=author,
    use_matlab_names=True,
)

# plot the sequence
if plot_sequence:
    seq.plot()
