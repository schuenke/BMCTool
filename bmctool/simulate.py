"""
simulate.py
    Script to run the BMCTool simulation based on a seq-file and a *.yaml config file.
"""

from typing import Union
from pathlib import Path
from bmctool.bmc_tool import BMCTool
from bmctool.utils.eval import plot_z
from bmctool.set_params import load_params


def simulate(config_file: Union[str, Path],
             seq_file: Union[str, Path],
             show_plot: bool = False,
             **kwargs) \
        -> BMCTool:
    """
    Function to run the BMCTool simulation based on a seq-file and a *.yaml config file..
    :param config_file: Path of the config file (can be of type str or Path)
    :param seq_file: Path of the seq file (can be of type str or Path)
    :param show_plot: flag to switch plotting option on/off
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f'File {config_file} not found.')

    if not Path(seq_file).exists():
        raise FileNotFoundError(f'File {seq_file} not found.')

    # load config file(s)
    sim_params = load_params(config_file)

    # create BMCTool object and run simulation
    sim = BMCTool(sim_params, seq_file, **kwargs)
    sim.run()

    if show_plot:
        if 'x' in kwargs:
            x = kwargs.pop('x')
            _, y = sim.get_zspec()
        else:
            x, y = sim.get_zspec()

        plot_z(y=y,
               x=x,
               **kwargs)

    return sim


def sim_example():
    seq_file = Path(__file__).parent / 'library' / 'seq-library' / 'WASABI.seq'
    config_file = Path(__file__).parent / 'library' / 'sim-library' / 'config_wasabi.yaml'

    simulate(config_file=config_file,
                   seq_file=seq_file,
                   show_plot=True,
                   title='WASABI example spectrum')
