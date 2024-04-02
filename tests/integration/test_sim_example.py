from pathlib import Path

import numpy as np
from bmctool.simulation._simulate import sim_example


def test_sim_example():
    """Test that the example simulation result and reference results match."""
    # load reference results from txt file
    ref = np.loadtxt(Path(__file__).parent / 'example_results.txt')

    # run example simulation
    _, result = sim_example(show_plot=False)

    # compare results
    assert np.allclose(result, ref, atol=1e-12)
