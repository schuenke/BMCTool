"""
simulate.py
    Script to run the BMCTool simulation based on the defined parameters.
    You can adapt parameters in param_configs.py or use a standard CEST setting as defined in standard_cest_params.py.
"""
from os import path
from pathlib import Path
from bmctool.sim.bmc_tool import BMCTool
from bmctool.sim.utils.eval import plot_z
from bmctool.sim.set_params import load_params


def simulate(config_file: str = None, seq_file:str = None):

    if config_file is None:
        config_file = Path(path.dirname(__file__)) / 'library' / 'sim-library' / 'config_wasabi.yaml'

    if seq_file is None:
        seq_file = Path(path.dirname(__file__)) / 'library' / 'seq-library' / 'WASABI.seq'

    # load config file(s)
    sim_params = load_params(config_file)

    # create BMCToll object and run simulation
    Sim = BMCTool(sim_params, seq_file)
    Sim.run()

    # extract and plot z-spectrum
    offsets, mz = Sim.get_zspec()
    fig = plot_z(mz=mz, offsets=offsets, invert_ax=True, plot_mtr_asym=True)

if __name__ == '__main__':
    simulate()
