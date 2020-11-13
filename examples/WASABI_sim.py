# Standard library imports
import sys
import os

# Add path
sys.path.append(os.path.join('D:\\','OneDrive','Dokumente','Python','BMCSim'))

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from BMCtool import BMCTool

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# simulation settings
n_timesteps = 1  # number of timesteps for each RF pulse
track = False  # track the magnetization trajectory during pulses/delays ?!
par_calc = True  # calculate all offsets in parallel instead of subsequently

# experimental settings
B0 = 3  # B0 in T
B1 = 1.25e-6*B0  # B1 in µT (for WASABI use 1.25e-6*B0)
n_p = 1  # number of saturation pulses
tp = 0.015/B0  # pulse duration in s (for WASABI use 0.015/B0)
td = 0  # delay between pulses in s
shape = 'CW'
trec = 5  # recovery/delay time between offsets in s
offset = 2  # max offset in ppm
n_offsets = 101  # number of offsets

# pool settings
n_pools = 1

T1n = np.array([2])
T2n = np.array([0.2])
fn = np.array([1])
dwn = np.array([0])
kn = np.array([0])

# calculate offsets
offsets = np.linspace(-offset, offset, n_offsets)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# create Sim object
Sim = BMCTool(b0=B0,
              n_pools=n_pools,
              t1_values=T1n,
              t2_values=T2n,
              poolsizes=fn,
              resonances=dwn,
              exchange_rates=kn,
              track=track)

if par_calc:
    # set initial magnetization
    Sim.set_M(offsets.shape[0])
 
    # simulate recovery
    Sim.solve(offsets, 0, trec, n_timesteps, shape=shape)

    # simulate saturation pulse train
    for n in range(n_p):
        # delay between pulses (not for the first pulse)
        if n != 0 and td != 0:
            Sim.solve(offsets, b1=0, pulse_dur=td, steps=int(td / tp * n_timesteps))
        # saturation
        Sim.solve(offsets, b1=B1, pulse_dur=tp, steps=int(n_timesteps), shape=shape)
        
    M = Sim.M_
else:
    # create array for magnetization values
    M = np.zeros([len(offsets), 3*n_pools, 1])
    
    # set initial magnetization
    Sim.set_M(1)

    for i in range(offsets.shape[0]):
        # simulate recovery
        Sim.solve(offsets[i], 0, trec, n_timesteps, shape='CW')

        # simulate saturation pulse train
        for n in range(n_p):
            if n != 0 and td != 0:
                Sim.solve(offsets[i], b1=0, pulse_dur=td, steps=int(td / tp * n_timesteps))
            Sim.solve(offsets[i], b1=B1, pulse_dur=tp, steps=int(n_timesteps), shape=shape)
        
        # write magnetization in array
        M[i,] = Sim.M_


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                  
fig, ax = plt.subplots(figsize=(12,9))

ax.plot(offsets, M[:,2,0], marker='o', linestyle='--', linewidth=2, color='black')
ax.set_xlabel('frequency offset [ppm]', fontsize=20)
ax.set_ylabel('normalized signal', fontsize=20)
ax.set_ylim([-0.1,1])
ax.invert_xaxis()
ax.grid()
plt.show()