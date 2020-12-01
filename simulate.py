"""
simulate.py
    Script to run the BMCTool simulation based on the defined parameters.
    You can adapt parameters in param_configs.py or use a standard CEST setting as defined in standard_cest_params.py.
"""
from sim.bmc_tool import BMCTool
from sim.utils.eval import plot_z
from sim.set_params import load_params

# set necessary file paths:
config_file = 'library/sim-library/config_wasabi.yaml'
seq_file = 'library/seq-library/T1prep_HSsat.seq'

# load config file(s)
sim_params = load_params(config_file)

# create BMCToll object and run simulation
Sim = BMCTool(sim_params, seq_file)
Sim.run()

# extract and plot z-spectrum
offsets, mz = Sim.get_zspec()
fig = plot_z(mz=mz, offsets=offsets, invert_ax=True, plot_mtr_asym=True)
