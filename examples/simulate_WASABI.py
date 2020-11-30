"""
simulate_WASABI.py
    Script to run the BMCTool simulation for WASABI sequence.
"""
from sim.bmc_tool import BMCTool
from sim.utils.eval import plot_z
from sim.set_params import load_params

# set necessary file paths:
config_file = '../library/sim-library/config_wasabi.yaml'
seq_file = '../library/seq-library/WASABI.seq'

# load config file
sim_params = load_params(config_file)
sim_params.update_options(par_calc=True)

# create BMCToll object and run simulation
Sim = BMCTool(sim_params, seq_file)
Sim.run()

# extract and plot z-spectrum
offsets, mz = Sim.get_zspec()
fig = plot_z(mz=mz, offsets=offsets, invert_ax=True, plot_mtr_asym=False)
