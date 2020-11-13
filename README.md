# Bloch-McConnell (BMC) Simulation Tool

## __required settings:__
### simulation settings
- **n_timesteps**(int): number of timesteps for each RF pulse
- **track**(bool): track the magnetization trajectory during pulses/delays
- **par_calc**(bool): calculate all offsets in parallel instead of subsequently

### experimental settings
- **B0**(float): field strength (B0) in Tesla
- **B1**(float): rf amplitude (B1) in µT (for WASABI use 1.25e-6*B0)
- **n_p**(int): number of (saturation) pulses
- **td**(float): delay between (saturation) pulses in s
- **tp**(float): (saturation) pulse duration in s (for WASABI use 0.015/B0)
- **shape**(string): shape of (saturation) pulses (defined in 'PulseShapes.py')
- **trec**(float): recovery/delay time between offsets/measurements in s
- **offsets**(np.array): frequency offset values in ppm

### pool settings
- **n_pools**(int): number of (CEST) pools including the bulk pool
- **T1n**(np.array): T1 of individual pools in s (1st entry = bulk pool)
- **T2n**(np.array): T2 of individual pools in s
- **fn**(np.array): relative pool sizes of individual pools. (1st entry should be 1)
- **dwn**(np.array): chemical shifts of individual pools in ppm. (1st entry should be 0)
- **kn**(np.array): exchange rates of individual pools in Hz. (1st entry is unused)

## __example code (for par_calc = True):__
```python

# create Sim object
Sim = BMCTool(B0=B0,
              n_pools=n_pools, 
              t1_values=T1n, 
              t2_values=T2n, 
              poolsizes=fn, 
              resonances=dwn, 
              exchange_rates=kn, 
              track=track)

# set initial magnetization
Sim.set_M(offsets.shape[0])
 
# simulate recovery
Sim.solve(offsets, 0, trec, n_timesteps, shape=shape)

# simulate saturation pulse train
for n in range(n_p):
    # delay between pulses (not for the first pulse)
    if n != 0 and td != 0:
        Sim.solve(offsets, b1=0, pulse_dur=td, steps=int(td/tp*n_timesteps))
    # saturation
    Sim.solve(offsets, b1=B1, pulse_dur=tp, steps=int(n_timesteps), shape=shape')
        
M = Sim.M_
```

