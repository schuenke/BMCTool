from pathlib import Path

import numpy as np
from bmctool.simulation import BMCSim
from bmctool.simulation.simulate import sim_example


def test_sim_example():
    """Test that the example simulation result and reference results match."""
    # load reference results from txt file
    ref = np.loadtxt(Path(__file__).parent / 'example_results.txt')

    # run example simulation
    _, result = sim_example(show_plot=False)

    # compare results
    assert np.allclose(result, ref, atol=1e-12)


def test_sim_from_fixtures(valid_parameters_object, valid_sequence_object):
    """Test that a simulation can be run using the provided fixtures."""
    sim = BMCSim(valid_parameters_object, valid_sequence_object)
    sim.run()
    offsets, spec = sim.get_zspec()

    assert len(offsets) == 11
    assert len(spec) == 11
