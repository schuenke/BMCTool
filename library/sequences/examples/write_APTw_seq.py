"""
Script to output a seq file for an APTw protocol
source: https://cest-sources.org/doku.php?id=standard_cest_protocols
APTw_1 : APT-weighted, low DC, t_sat=1.8s (//GLINT//)
     pulse shape = Gaussian
     B1 = 2.22 uT
     n = 20
     t_p = 50 ms
     t_d = 40 ms
     DC = 0.55 and t_sat = n*(t_p+t_d) = 1.8 s
     T_rec = 2.4/12 s (saturated/M0)
"""

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.opts import Opts
from sim.utils.seq import convert_seq_12_to_pseudo_13

seq = Sequence()

offset_range = 10  # [ppm]
num_offsets = 41  # 100  # number of measurements (not including M0)
run_m0_scan = True  # if you want an M0 scan at the beginning
t_rec = 2.4  # recovery time between scans [s]
m0_t_rec = 12  # recovery time before m0 scan [s]
sat_b1 = 2.22  # mean sat pulse b1 [uT]
t_p = 50e-3  # sat pulse duration [s]
t_d = 40e-3  # delay between pulses [s]
n_pulses = 20  # number of sat pulses per measurement
b0 = 3  # B0 [T]
spoiling = 1  # 0=no spoiling, 1=before readout, Gradient in x,y,z

seq_filename = 'example_APTw.seq'  # filename

# scanner limits
sys = Opts(max_grad=40, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', rf_ringdown_time=30e-6, rf_dead_time=100e-6,
           rf_raster_time=1e-6)
gamma = sys.gamma * 1e-6

# scanner events
# sat pulse
flip_angle_sat = sat_b1 * gamma * 2 * np.pi * t_p  # rad
rf_sat, _, _ = make_gauss_pulse(flip_angle=flip_angle_sat, duration=t_p, system=sys, time_bw_product=3)

# spoilers
spoil_amp = 0.8 * sys.max_grad  # Hz/m
spoil_dur = 4500e-6  # s
gx_spoil, gy_spoil, gz_spoil = [make_trapezoid(channel=c, system=sys, amplitude=spoil_amp, duration=spoil_dur)
                                for c in ['x', 'y', 'z']]

# pseudo adc (not played out)
pseudo_adc = make_adc(num_samples=1, duration=1e-3)

# loop through z spectrum offsets
offsets_ppm = np.linspace(-offset_range, offset_range, num_offsets)
offsets = offsets_ppm * gamma * b0

if run_m0_scan:
    seq.add_block(make_delay(m0_t_rec))
    seq.add_block(pseudo_adc)

# loop through offsets and set pulses and delays
for o in offsets:
    # take care of phase accumulation during off-res pulse
    accum_phase = 0
    seq.add_block(make_delay(t_rec))  # recovery time
    rf_sat.freq_offset = o
    for n in range(n_pulses):
        rf_sat.phase_offset = accum_phase
        seq.add_block(rf_sat)
        accum_phase = np.mod(accum_phase + o * 2 * np.pi * np.where(np.abs(rf_sat.signal) > 0)[0].size * 1e-6, 2 * np.pi)
        if n < n_pulses-1:
            seq.add_block(make_delay(t_d))
    print(np.where(offsets == o)[0][0], '/', len(offsets), ': offset ', o)
    if spoiling:
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)
    seq.add_block(pseudo_adc)

seq.set_definition('offsets_ppm', offsets_ppm)
seq.set_definition('run_m0_scan', str(run_m0_scan))

# plot the sequence
# seq.plot()
print(seq.shape_library)
seq.write(seq_filename)

# convert to pseudo version 1.3
convert_seq_12_to_pseudo_13(seq_filename)
