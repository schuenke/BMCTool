"""
simulate_T1prep_HSsat.py
    Script to run the BMCTool simulation for an adiabatic T1 saturation recovery sequence.
"""
from sim.bmc_tool import BMCTool
from sim.utils.eval import plot_z
from sim.set_params import load_params

# set necessary file paths:
config_file = '../library/sim-library/config_wasabiti.yaml'
seq_file = '../library/seq-library/T1prep_HSsat.seq'

# load config file
sim_params = load_params(config_file)

# ensure parallel calculation (par_calc) is disabled
sim_params.update_options(par_calc=False)

# create BMCToll object and run simulation
Sim = BMCTool(sim_params, seq_file)
Sim.run()

# extract and plot spectrum
_, mz = Sim.get_zspec()
TI_times = Sim.seq.definitions['TI_times']
fig = plot_z(mz=mz, offsets=TI_times, invert_ax=False)
