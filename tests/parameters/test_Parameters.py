from copy import deepcopy

import pytest
from bmctool.parameters import Parameters


def test_properties_raise_error(valid_parameters_object):
    """Test that m_vec and mz_loc raise an error if water_pool does not exist."""
    p = deepcopy(valid_parameters_object)
    del p.water_pool
    with pytest.raises(Exception, match='No water pool defined.'):
        _ = p.mz_loc

    with pytest.raises(Exception, match='No water pool defined.'):
        _ = p.m_vec


def test_init_from_valid_dict(valid_config_dict):
    """Test initialization of the Parameters class from a valid dictionary."""
    p = Parameters.from_dict(valid_config_dict)

    # assert that the attributes are set correctly
    assert p.system.b0 == valid_config_dict['b0']
    assert p.system.gamma == valid_config_dict['gamma']
    assert p.system.b0_inhom == valid_config_dict['b0_inhom']
    assert p.system.rel_b1 == valid_config_dict['rel_b1']

    assert p.options.verbose == valid_config_dict['verbose']
    assert p.options.reset_init_mag == valid_config_dict['reset_init_mag']
    assert p.options.scale == valid_config_dict['scale']
    assert p.options.max_pulse_samples == valid_config_dict['max_pulse_samples']

    assert p.water_pool.r1 == valid_config_dict['water_pool']['r1']
    assert p.water_pool.r2 == valid_config_dict['water_pool']['r2']
    assert p.water_pool.f == valid_config_dict['water_pool']['f']

    assert p.cest_pools[0].f == valid_config_dict['cest_pool']['cest_1']['f']
    assert p.cest_pools[0].t1 == valid_config_dict['cest_pool']['cest_1']['t1']
    assert p.cest_pools[0].t2 == valid_config_dict['cest_pool']['cest_1']['t2']
    assert p.cest_pools[0].k == valid_config_dict['cest_pool']['cest_1']['k']
    assert p.cest_pools[0].dw == valid_config_dict['cest_pool']['cest_1']['dw']

    assert p.cest_pools[1].f == valid_config_dict['cest_pool']['cest_2']['f']
    assert p.cest_pools[1].t1 == valid_config_dict['cest_pool']['cest_2']['t1']
    assert p.cest_pools[1].t2 == valid_config_dict['cest_pool']['cest_2']['t2']
    assert p.cest_pools[1].k == valid_config_dict['cest_pool']['cest_2']['k']
    assert p.cest_pools[1].dw == valid_config_dict['cest_pool']['cest_2']['dw']


def test_init_from_valid_file(valid_config_dict, valid_yaml_config_file):
    """Test initialization of the Parameters class from a valid YAML config file."""
    p = Parameters.from_yaml(valid_yaml_config_file)
    q = Parameters.from_dict(valid_config_dict)

    assert p == q


def test_init_from_alternative_param_names(valid_config_dict):
    """Test initialization of the Parameters class from a dictionary with alternative parameter names."""
    # create copy of valid_config_dict and rename some parameters
    valid_config_dict_alt_names = valid_config_dict.copy()
    valid_config_dict_alt_names['relb1'] = valid_config_dict_alt_names.pop('rel_b1')
    valid_config_dict_alt_names['b0_inhomogeneity'] = valid_config_dict_alt_names.pop('b0_inhom')

    p = Parameters.from_dict(valid_config_dict_alt_names)
    q = Parameters.from_dict(valid_config_dict)

    assert p == q


def test_export_to_yaml(valid_parameters_object, empty_config_file):
    """Test that the Parameters class can be exported to a YAML config file."""
    p = deepcopy(valid_parameters_object)

    # export Parameters object to YAML config file
    p.to_yaml(empty_config_file)

    # create Parameters object from the exported config file
    q = Parameters.from_yaml(empty_config_file)

    # ensure that both objects are equal
    assert p == q


@pytest.mark.parametrize(
    ('parameter', 'value'),
    [
        ('t1', 3.0),
        ('t1', '5'),
        ('r1', 0.5),
        ('f', 0.5),
    ],
)
def test_update_water_pool_parameter(valid_parameters_object, parameter, value):
    """Test that the water pool parameters are updated correctly."""
    p = deepcopy(valid_parameters_object)

    # call p.update_water_pool() with the parameter and value
    p.update_water_pool(**{parameter: value})

    # assert that the attribute was updated correctly
    assert getattr(p.water_pool, parameter) == float(value)


@pytest.mark.parametrize(
    ('parameter', 'value'),
    [
        ('f', 0.5),
        ('t1', 3.0),
        ('t2', 5.0),
        ('k', 100.0),
        ('dw', 3.0),
    ],
)
def test_update_mt_pool_parameter(valid_parameters_object, parameter, value):
    """Test that the MT pool parameters are updated correctly."""
    p = deepcopy(valid_parameters_object)

    # call p.update_mt_pool() with the parameter and value
    p.update_mt_pool(**{parameter: value})

    # assert that the attribute was updated correctly
    assert getattr(p.mt_pool, parameter) == value


@pytest.mark.parametrize(
    ('parameter', 'value', 'pool_idx'),
    [
        ('f', 0.5, 0),
        ('t1', 3.0, 0),
        ('t2', 5.0, 0),
        ('k', 100.0, 0),
        ('dw', 3.0, 0),
        ('f', 0.5, 1),
    ],
)
def test_update_cest_pool_parameter(valid_parameters_object, valid_cest_pool_object, parameter, value, pool_idx):
    """Test that the MT pool parameters are updated correctly."""
    p = deepcopy(valid_parameters_object)
    p.add_cest_pool(valid_cest_pool_object)  # add 2nd identical CEST pool

    # call p.update_cest_pool() with the parameter and value
    p.update_cest_pool(pool_idx, **{parameter: value})

    # assert that the attribute was updated correctly
    assert getattr(p.cest_pools[pool_idx], parameter) == value


@pytest.mark.parametrize(
    ('parameter', 'value'),
    [
        ('verbose', True),
        ('reset_init_mag', False),
        ('scale', 0.0),
        ('max_pulse_samples', 123),
    ],
)
def test_update_options(valid_parameters_object, parameter, value):
    """Test that the options are updated correctly."""
    p = deepcopy(valid_parameters_object)

    # call p.update_options() with a new value for verbose
    p.update_options(**{parameter: value})

    # assert that the attribute was updated correctly
    assert getattr(p.options, parameter) == value


@pytest.mark.parametrize(
    ('parameter', 'value'),
    [
        ('b0', 7.0),
        ('gamma', 12.3456),
        ('b0_inhom', 0.1),
        ('rel_b1', 1.1),
    ],
)
def test_update_system_parameter(valid_parameters_object, parameter, value):
    """Test that the system parameters are updated correctly."""
    p = deepcopy(valid_parameters_object)

    # call p.update_system() with the parameter and value
    p.update_system(**{parameter: value})

    # assert that the attribute was updated correctly
    assert getattr(p.system, parameter) == value


def test_equality(
    valid_parameters_object,
    valid_water_pool_object,
    valid_cest_pool_object,
    valid_mt_pool_object,
    valid_options_object,
    valid_system_object,
):
    """Test that Options instances are equal if their attributes are equal."""
    a = Parameters(
        water_pool=valid_water_pool_object,
        cest_pools=[valid_cest_pool_object],
        mt_pool=valid_mt_pool_object,
        options=valid_options_object,
        system=valid_system_object,
    )
    b = deepcopy(valid_parameters_object)
    b.options.max_pulse_samples = 10
    c = deepcopy(valid_parameters_object)
    c.water_pool.r1 = 2.0

    assert a == valid_parameters_object
    assert a != b
    assert a != c
    assert a != 'not a Parameters object'
