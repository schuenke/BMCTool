"""
WASABI_sim.py
    Script to run a WASABI simulation using the BMCTool.
"""
from sim.bmc_tool import BMCTool
from sim.utils.eval import plot_z
from sim.set_params import load_params

# set WASABI seq and config files
sample_file = '../library/param_configs/wasabi_sample_params.yaml'
experimental_file = '../library/param_configs/wasabi_experimental_params.yaml'
seq_file = '../library/sequences/examples/example_wasabi.seq'

# load config files and print settings
sim_params = load_params(sample_file, experimental_file)

# create BMCToll object and run simulation
Sim = BMCTool(sim_params, seq_file)
Sim.run()

# extract and plot z-spectrum
offsets, mz = Sim.get_zspec()
fig = plot_z(mz=mz, offsets=offsets, plot_mtr_asym=False)
