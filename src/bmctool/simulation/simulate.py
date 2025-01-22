"""Functions to run BMCTool simulations based on given seq-file and config file."""

from pathlib import Path

import numpy as np
from pypulseq import Sequence  # type: ignore

from bmctool.parameters.Parameters import Parameters
from bmctool.simulation.BMCSim import BMCSim
from bmctool.utils.eval import plot_z


def simulate(
    config: str | Path | Parameters,
    seq: str | Path | Sequence,
    show_plot: bool = False,
    **kwargs,
) -> BMCSim:
    """Run BMCTool simulation based on given seq-file and config file.

    Parameters
    ----------
    config
        Path to the YAML config file or Parameters object
    seq
        Path to the Pulseq sequence file or PyPulseq Sequence object
    show_plot, optional
        Flag to activate plotting of simulated data, by default False
    **kwargs
        Additional keyword arguments passed to the plot_z function

    Returns
    -------
    BMCTool
        BMCTool object containing the simulation results.

    Raises
    ------
    FileNotFoundError
        If the config_file or seq_file not found.
    """
    if isinstance(config, Parameters):
        sim_params = config
    else:
        if not Path(config).exists():
            raise FileNotFoundError(f'File {config} not found.')
        # load config file(s)
        sim_params = Parameters.from_yaml(config)

    if not isinstance(seq, Sequence) and not Path(seq).exists():
        raise FileNotFoundError(f'File {seq} not found.')

    # create BMCTool object and run simulation
    sim = BMCSim(sim_params, seq)
    sim.run()

    if show_plot:
        if 'offsets' in kwargs:
            offsets = kwargs.pop('offsets')
            _, m_z = sim.get_zspec()
        else:
            offsets, m_z = sim.get_zspec()

        plot_z(m_z=m_z, offsets=offsets, **kwargs)

    return sim


def sim_example(show_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Run an example simulation using included config and seq files.

    Parameters
    ----------
    show_plot, optional
        Flag to activate plotting of simulated data, by default True
    """
    seq_file = Path(__file__).parent.parent / 'library' / 'seq-library' / 'WASABI.seq'
    config_file = Path(__file__).parent.parent / 'library' / 'sim-library' / 'config_1pool.yaml'

    sim = simulate(
        config=config_file,
        seq=seq_file,
        show_plot=show_plot,
        title='WASABI example spectrum',
        normalize=True,
    )

    offsets, mz = sim.get_zspec(return_abs=False)
    return offsets, mz


if __name__ == '__main__':
    sim_example()
