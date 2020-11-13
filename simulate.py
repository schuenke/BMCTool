"""
simulate.py
    Script to run the BMCTool simulation based on the defined parameters.
    You can adapt parameters in param_configs.py or use a standard CEST setting as defined in standard_cest_params.py.
"""
from sim.bmc_tool import BMCTool
from utils.sim.eval import plot_z
from sim.set_params import load_params

# set necessary file paths:
sample_file = 'dictionary/param_configs/sample_params.yaml'
experimental_file = 'dictionary/param_configs/experimental_params.yaml'
seq_file = 'dictionary/sequences/examples/example_test.seq'

# load config files and print settings
sim_params = load_params(sample_file, experimental_file)

# create BMCToll object and run simulation
Sim = BMCTool(sim_params, seq_file)
Sim.run(par_calc=sim_params.options['par_calc'])

# extract and plot z-spectrum
offsets, mz = Sim.get_zspec()
fig = plot_z(mz=mz, offsets=offsets, plot_mtr_asym=True)
