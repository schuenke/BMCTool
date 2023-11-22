"""simulate.py Script to run the BMCTool simulation based on a seq-file and a
*.yaml config file."""

from pathlib import Path

from src.bmctool.bmc_tool import BMCTool
from src.bmctool.set_params import load_params
from src.bmctool.utils.eval import plot_z


def simulate(config_file: str | Path, seq_file: str | Path, show_plot: bool = False, **kwargs) -> BMCTool:
    """Simulate Run BMCTool simulation based on given seq-file and config file.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the config file.
    seq_file : Union[str, Path]
        Path to the seq-file.
    show_plot : bool, optional
        Flag to activate plotting of simulated data, by default False

    Returns
    -------
    BMCTool
        BMCTool object containing the simulation results.

    Raises
    ------
    FileNotFoundError
        If the config_file or seq_file is not found.
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
        if 'offsets' in kwargs:
            offsets = kwargs.pop('offsets')
            _, m_z = sim.get_zspec()
        else:
            offsets, m_z = sim.get_zspec()

        plot_z(m_z=m_z, offsets=offsets, **kwargs)

    return sim


def sim_example() -> None:
    """Function to run an example WASABI simulation."""
    seq_file = Path(__file__).parent / 'library' / 'seq-library' / 'WASABI.seq'
    config_file = Path(__file__).parent / 'library' / 'sim-library' / 'config_wasabi.yaml'

    simulate(
        config_file=config_file, seq_file=seq_file, show_plot=True, title='WASABI example spectrum', normalize=True
    )


if __name__ == '__main__':
    sim_example()
