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

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.opts import Opts
from sim.utils.seq.conversion import convert_seq_12_to_pseudo_13

seq = Sequence()

ti_range = 2  # [s]
num_ti = 21  # number of measurements (not including M0)
run_m0_scan = False  # if you want an M0 scan at the beginning
b0 = 3  # B0 [T]
spoiling = 1  # 0=no spoiling, 1=before readout, Gradient in x,y,z
spoil_delay = 0.1e-3
initial_delay = 100e-3
reset_init_mag = True
# adc_delay = 50e-6

seq_filename = '../../../../../seq_misc/writeSeq/sequences/T1_prep.seq'  # filename

# scanner limits
sys = Opts(max_grad=40, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', rf_ringdown_time=30e-6, rf_dead_time=100e-6,
           rf_raster_time=1e-6)
gamma = sys.gamma * 1e-6

# scanner events

# spoilers
spoil_amp = 0.8 * sys.max_grad  # Hz/m
spoil_amp1 = -0.7 * sys.max_grad  # Hz/m
spoil_amp2 = 0.6 * sys.max_grad  # Hz/m
spoil_dur = 5500e-6  # s
rise_time = 1e-3
gx_spoil0, gy_spoil0, gz_spoil0 = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp, duration=spoil_dur,
                                               rise_time=rise_time) for c in ['x', 'y', 'z']]
gx_spoil1, gy_spoil1, gz_spoil1 = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp1, duration=spoil_dur,
                                               rise_time=rise_time) for c in ['x', 'y', 'z']]
gx_spoil2, gy_spoil2, gz_spoil2 = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp2, duration=spoil_dur,
                                               rise_time=rise_time) for c in ['x', 'y', 'z']]

# 90 degree pulse
flip_angle_t1 = 90 * np.pi / 180
t1_dur = 2.5e-3
t1_rf, _ = make_block_pulse(flip_angle=flip_angle_t1, duration=t1_dur, system=sys)
# t1_rf, _, _ = make_sinc_pulse(flip_angle=flip_angle_t1, duration=t1_dur, system=sys, time_bw_product=2, apodization=0.15)

# pseudo adc (not played out)
pseudo_adc = make_adc(num_samples=1, duration=1e-3)

ti_vec = np.flip([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 6000, 8000, 10000, 12000]) * 1e-3 # [s]

# loop through offsets and set pulses and delays
for ti in ti_vec:
    seq.add_block(make_delay(initial_delay))
    if reset_init_mag:
        for i in range(42):
            seq.add_block(t1_rf)
            if i % 3 == 0:
                seq.add_block(gx_spoil0, gy_spoil1, gz_spoil2)
            elif i % 2 == 0:
                seq.add_block(gx_spoil2, gy_spoil0, gz_spoil1)
            else:
                seq.add_block(gx_spoil1, gy_spoil2, gz_spoil0)
            seq.add_block(make_delay(spoil_delay))
    seq.add_block(make_delay(ti))  # recovery time
    if spoiling:
        seq.add_block(gx_spoil0, gy_spoil0, gz_spoil0)
        seq.add_block(make_delay(spoil_delay))
    # seq.add_block(make_delay(adc_delay))
    seq.add_block(pseudo_adc)

seq.set_definition('offsets_ppm', np.zeros(ti_vec.shape))
seq.set_definition('run_m0_scan', str(run_m0_scan))
seq.set_definition('ti_ms', ti_vec*1e3)

# plot the sequence
# seq.plot()
print(seq.shape_library)
seq.write(seq_filename)
# convert to pseudo version 1.3
convert_seq_12_to_pseudo_13(seq_filename)
