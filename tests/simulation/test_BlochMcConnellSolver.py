from copy import deepcopy

import numpy as np
import pytest
from bmctool.parameters import Parameters
from bmctool.simulation import BlochMcConnellSolver


def test_init_from_valid_dict(valid_parameters_object):
    """Test initializatim from a Parameters object and n_offsets."""
    params = deepcopy(valid_parameters_object)
    solver = BlochMcConnellSolver(params=params, n_offsets=100)

    assert solver is not None
    assert solver.params == params


def test_correct_shape(valid_config_dict):
    """Test that array a and c have the correct shape."""
    params = Parameters.from_dict(valid_config_dict)
    solver = BlochMcConnellSolver(params=params, n_offsets=100)

    # check that size attribute is correct
    assert solver.size == params.m_vec.size

    # check that arr_a has the correct shape
    assert solver.arr_a.shape == (9, 9)  # valid_config has 1 water, 2 cest pools

    # check hat arr_c has the correct shape
    assert solver.arr_c.shape == (9,)  # valid_config has 1 water, 2 cest pools


def test_correct_attributes(valid_config_dict):
    """Test that all other attributes are correct."""
    params = Parameters.from_dict(valid_config_dict)
    solver = BlochMcConnellSolver(params=params, n_offsets=100)

    # assert that the number of offsets is correct
    assert solver.n_offsets == 100

    # assert number of cest pools is correct
    assert solver.n_pools == 2

    # assert that w0 is correct
    assert solver.w0 == params.system.b0 * params.system.gamma

    # assert that dw0 is correct
    assert solver.dw0 == params.system.b0_inhom


@pytest.mark.parametrize(
    ('parameter', 'value'),
    [
        ('b0', 1.0),
        ('b0', 2.0),
        ('b0_inhom', 1.0),
        ('b0_inhom', 2.0),
    ],
)
def test_update_params(valid_parameters_object, parameter, value):
    """Test that update_params() works correctly."""
    params = deepcopy(valid_parameters_object)
    solver = BlochMcConnellSolver(params=params, n_offsets=100)

    # change field strength in Parameters object
    params.update_system(**{parameter: value})

    # update solver with new parameters
    solver.update_params(params=params)

    # assert that changed parameters are correct
    assert solver.w0 == params.system.b0 * params.system.gamma
    assert solver.dw0 == solver.w0 * params.system.b0_inhom


def test_solve_equation_correct_shape(valid_config_dict):
    """Test that solve_equation() returns an array with the correct shape."""
    params = Parameters.from_dict(valid_config_dict)
    solver = BlochMcConnellSolver(params=params, n_offsets=100)

    # get initial magnetization in required shape
    mag = params.m_vec[:]

    # solve equation
    res = solver.solve_equation(mag=mag, dtp=1e-3)

    # assert that res has the correct shape
    assert res.shape == (9,)


@pytest.mark.parametrize(
    ('time', 'value'),
    [
        (1, 0.6321205588),
        (2, 0.8646647167),
        (10, 0.9999546000),
        (100, 1.0),
    ],
)
def test_solve_equation_relaxation_from_zero(valid_config_dict_only_water, time, value):
    """Test that solve_equation() returns an array with the correct shape."""
    params = Parameters.from_dict(valid_config_dict_only_water)
    solver = BlochMcConnellSolver(params=params, n_offsets=100)

    # set water z-magnetization to zero before relaxation
    mag = params.m_vec[:]
    mag[params.mz_loc] = 0.0

    # solve equation for current relaxation time
    res = np.squeeze(solver.solve_equation(mag=mag, dtp=time))

    # assert that water z-magnetization matches expected value
    assert np.isclose(res[params.mz_loc], value, atol=1e-9)
