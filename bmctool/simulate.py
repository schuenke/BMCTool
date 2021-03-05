"""
simulate.py
    Script to run the BMCTool simulation based on a seq-file and a *.yaml config file.
"""

from os import path
from pathlib import Path
from bmctool.bmc_tool import BMCTool
from bmctool.utils.eval import plot_z
from bmctool.set_params import load_params


def simulate(config_file: (str, Path) = None, seq_file: (str, Path) = None):
    """
    Function to run the BMCTool simulation based on a seq-file and a *.yaml config file..
    :param config_file: Path of the config file (can be of type str or Path)
    :param seq_file: Path of the seq file (can be of type str or Path)
    """

    if config_file is None:
        config_file = Path(path.dirname(__file__)) / 'library' / 'sim-library' / 'config_wasabiti.yaml'

    if seq_file is None:
        seq_file = Path(path.dirname(__file__)) / 'library' / 'seq-library' / 'WASABI.seq'

    # load config file(s)
    sim_params = load_params(config_file)

    # create BMCTool object and run simulation
    sim = BMCTool(sim_params, seq_file)
    sim.run()

    # extract and plot z-spectrum
    offsets, mz = sim.get_zspec()
    fig = plot_z(mz=mz, offsets=offsets, invert_ax=True, plot_mtr_asym=False, title='Example WASABI spectrum')


if __name__ == '__main__':
    simulate()
