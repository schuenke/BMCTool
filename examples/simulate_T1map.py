"""simulate_T1map.py Script to run the BMCTool simulation for a T1 mapping
sequence."""
from pathlib import Path

from bmctool.simulate import simulate
from bmctool.utils.seq.auxiliary import get_definition

# set necessary file paths:
config_file = Path(__file__).parent / 'library' / 'example_config.yaml'
seq_file = Path(__file__).parent / 'library' / 'T1map.seq'

# read recovery times from seq file definitions
TI = get_definition(seq_file, 'TI')

sim = simulate(
    config_file=config_file,
    seq_file=seq_file,
    show_plot=True,
    verbose=True,
    offsets=TI,
    x_label='TI [s]',
    invert_ax=False,
    title='Example T1 saturation recovery curve',
)
