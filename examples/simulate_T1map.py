"""
simulate_T1map.py
    Script to run the BMCTool simulation for a T1 mapping sequence.
"""
from pathlib import Path
from bmctool.bmc_tool import BMCTool
from bmctool.utils.eval import plot_z
from bmctool.set_params import load_params
from bmctool.utils.seq.auxiliary import get_definition

# set necessary file paths:
config_file = Path(__file__).parent / 'library' / 'example_config.yaml'
seq_file = Path(__file__).parent / 'library' / 'T1map.seq'

# load config file(s) and print settings
sim_params = load_params(config_file)
sim_params.print_settings()

# create BMCTool object and run simulation
sim = BMCTool(sim_params, seq_file)
sim.run()

# read recovery times from seq file definitions
TI = get_definition(seq_file, 'TI')

# extract and plot z-spectrum
offsets, mz = sim.get_zspec()
fig = plot_z(mz=mz,
             offsets=TI,
             invert_ax=False,
             plot_mtr_asym=False,
             title='Example T1 saturation recovery curve',
             x_label='TI [s]')

