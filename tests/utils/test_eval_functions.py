import numpy as np
import pytest
from bmctool.utils.eval import calc_mtr_asym
from bmctool.utils.eval import normalize_data
from bmctool.utils.eval import plot_z
from bmctool.utils.eval import split_data
from matplotlib.figure import Figure


def test_calc_mtr_asym():
    """Test the calc_mtr_asym function."""

    # simple symmetric data (odd number of points)
    offsets = np.array([-3, -2, -1, 0, 1, 2, 3])
    m_z = np.array([1, 2, 3, 4, 3, 2, 1])
    expected_asym = np.array([0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_almost_equal(calc_mtr_asym(m_z, offsets), expected_asym)

    # simple symmetric data (even number of points)
    offsets = np.array([-3, -2, -1, 1, 2, 3])
    m_z = np.array([1, 2, 3, 3, 2, 1])
    expected_asym = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_array_almost_equal(calc_mtr_asym(m_z, offsets), expected_asym)

    # symmetric offsets, asymmetric data
    offsets = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    m_z = np.array([1, 1, 0.5, 1, 1, 0, 1, 1, 1, 1, 1])
    expected_asym = np.array([0, 0, 0.5, 0, 0, 0, 0, 0, -0.5, 0, 0], dtype=float)
    np.testing.assert_array_almost_equal(calc_mtr_asym(m_z, offsets), expected_asym)

    # asymmetric offsets, asymmetric data
    offsets = np.array([-5, -4.5, -4, -3.5, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    m_z = np.array([1, 1, 1, 1, 0.5, 1, 1, 0, 1, 1, 1, 1, 1])
    expected_asym = np.array([0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, -0.5, 0, 0], dtype=float)
    np.testing.assert_array_almost_equal(calc_mtr_asym(m_z, offsets), expected_asym)

    # wrong dimensions
    with pytest.raises(ValueError, match='m_z and offsets must have the same dimensions.'):
        calc_mtr_asym(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))


@pytest.mark.parametrize(
    ('threshold', 'expected_norm_value'),
    [
        (5, 80),
        (-5, 80),
        (299, 80),
        (300, 80),
        (301, 1),
    ],
)
def test_normalize_data(threshold, expected_norm_value):
    """Test the normalize_data function with realistic data."""
    offsets = np.array([-300, -3, -2, -1, 0, 1, 2, 3])
    m_z = np.array([80, 40, 30, 20, 10, 20, 30, 40], dtype=float)
    if expected_norm_value != 1:
        expected_normalized_m_z = m_z[1:] / expected_norm_value
    else:
        expected_normalized_m_z = m_z

    normalized_m_z, _ = normalize_data(m_z, offsets, threshold)
    np.testing.assert_array_almost_equal(normalized_m_z, expected_normalized_m_z)


@pytest.mark.parametrize(
    ('m_z', 'offsets', 'threshold', 'expected_offsets', 'expected_data', 'expected_norm_data'),
    [
        # Test case 1: threshold is a single float
        (
            np.array([1, 2, 3, 4, 5]),  # m_z
            np.array([-1, 0, 1, 2, 3]),  # offsets
            1.5,  # threshold
            np.array([-1, 0, 1]),  # expected_offsets
            np.array([1, 2, 3]),  # expected_data
            np.array([4, 5]),  # expected_norm_data
        ),
        # Test case 2: threshold is a single integer
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([-1, 0, 1, 2, 3]),
            2,
            np.array([-1, 0, 1]),
            np.array([1, 2, 3]),
            np.array([4, 5]),
        ),
        # Test case 3: threshold is a list of two values
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([-3, -2, -1, 0, 1]),
            [-1.0, 1],
            np.array([0]),
            np.array([4]),
            np.array([1, 2, 3, 5]),
        ),
        # Test case 4: threshold is a numpy array of two values
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([-3, -2, -1, 0, 1]),
            np.array([-1.0, 1]),
            np.array([0]),
            np.array([4]),
            np.array([1, 2, 3, 5]),
        ),
        # Test case 5: normalization data is None
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([0, 1, 2, 3, 4]),
            10,
            np.array([0, 1, 2, 3, 4]),
            np.array([1, 2, 3, 4, 5]),
            None,
        ),
    ],
)
def test_split_data(m_z, offsets, threshold, expected_offsets, expected_data, expected_norm_data):
    result_offsets, result_data, result_norm = split_data(m_z, offsets, threshold)

    assert np.array_equal(result_offsets, expected_offsets), f'expected {expected_offsets}, got {result_offsets}'
    assert np.array_equal(result_data, expected_data), f'expected {expected_data}, got {result_data}'

    if expected_norm_data is None:
        assert result_norm is None, f'expected None, got {result_norm}'
    else:
        assert np.array_equal(result_norm, expected_norm_data), f'expected {expected_norm_data}, got {result_norm}'  # type: ignore


def test_split_data_raises():
    """Test the split_data function with invalid threshold values."""
    with pytest.raises(TypeError, match='Threshold of type'):
        split_data(np.array([1, 2, 3]), np.array([1, 2, 3, 4]), 'str')  # type: ignore


def test_plot_z():
    """Test the plot_z function."""

    # define offsets
    offsets = np.linspace(-6, 6, 13)
    m_z = np.array([1, 0.75, 0.5, 0.75, 1, 1, 0, 1, 1, 1, 1, 1, 1])

    # Test case 1: Basic test case with default parameters
    expected_fig = plot_z(m_z=m_z, show_plot=False)
    assert isinstance(expected_fig, Figure)

    # Test case 2: Test with custom offsets
    expected_fig = plot_z(m_z=m_z, offsets=offsets, show_plot=False)
    assert isinstance(expected_fig, Figure)

    # Test case 3: Test with norm=True, but no threshold
    normalize = True
    expected_fig = plot_z(m_z=m_z, offsets=offsets, normalize=normalize, show_plot=False)
    assert isinstance(expected_fig, Figure)

    # Test case 4: Test with normalization and custom threshold
    expected_fig = plot_z(m_z=m_z, offsets=offsets, normalize=normalize, norm_threshold=6, show_plot=False)
    assert isinstance(expected_fig, Figure)

    # Test case 5: Test without inverted x-axis
    expected_fig = plot_z(m_z=m_z, offsets=offsets, invert_ax=False, show_plot=False)
    assert isinstance(expected_fig, Figure)

    # Test case 6: Test with MTRasym plot
    expected_fig = plot_z(m_z=m_z, offsets=offsets, plot_mtr_asym=True, show_plot=False)
    assert isinstance(expected_fig, Figure)

    # Test case 7: Test with custom title, x_label, and y_label
    title = 'Custom Title'
    x_label = 'Custom X Label'
    y_label = 'Custom Y Label'
    expected_fig = plot_z(m_z=m_z, offsets=offsets, title=title, x_label=x_label, y_label=y_label, show_plot=False)
    assert isinstance(expected_fig, Figure)
