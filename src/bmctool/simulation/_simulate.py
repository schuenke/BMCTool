from __future__ import annotations

from pathlib import Path

from bmctool.parameters import Parameters
from bmctool.simulation import BMCSim
from bmctool.utils.eval import plot_z


def simulate(
    config_file: str | Path,
    seq_file: str | Path,
    show_plot: bool = False,
    **kwargs,
) -> BMCSim:
    """Run BMCTool simulation based on given seq-file and config file.

    Parameters
    ----------
    config_file
        Path to the YAML config file
    seq_file
        Path to the pulseq sequence file
    show_plot (optional)
        Flag to activate plotting of simulated data, by default False

    Returns
    -------
    BMCTool
        BMCTool object containing the simulation results.

    Raises
    ------
    FileNotFoundError
        If the config_file or seq_file not found.
    """

    if not Path(config_file).exists():
        raise FileNotFoundError(f'File {config_file} not found.')

    if not Path(seq_file).exists():
        raise FileNotFoundError(f'File {seq_file} not found.')

    # load config file(s)
    sim_params = Parameters.from_yaml(config_file)

    # create BMCTool object and run simulation
    sim = BMCSim(sim_params, seq_file, **kwargs)
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
    """Run an example simulation using included config and seq files."""

    seq_file = Path(__file__).parent.parent / 'library' / 'seq-library' / 'WASABI.seq'
    config_file = Path(__file__).parent.parent / 'library' / 'sim-library' / 'config_wasabi.yaml'

    simulate(
        config_file=config_file,
        seq_file=seq_file,
        show_plot=True,
        title='WASABI example spectrum',
        normalize=True,
    )


if __name__ == '__main__':
    sim_example()
