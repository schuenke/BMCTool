"""
simulate_WASABI.py
    Script to run the BMCTool simulation for WASABI sequence.
"""
from pathlib import Path
from bmctool.simulate import simulate

# set necessary file paths:
config_file = Path(__file__).parent / 'library' / 'example_config.yaml'
seq_file = Path(__file__).parent / 'library' / 'WASABI.seq'


sim = simulate(config_file=config_file,
               seq_file=seq_file,
               show_plot=True,
               verbose=False,
               title='Example WASABI spectrum')

