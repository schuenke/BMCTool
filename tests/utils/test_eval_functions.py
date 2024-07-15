import numpy as np
import pytest
from bmctool.utils.eval import calc_mtr_asym
from bmctool.utils.eval import normalize_data
from bmctool.utils.eval import split_data


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
