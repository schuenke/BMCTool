import pytest

from bmctool.parameters import Parameters
from tests.conftest import valid_config_dict
from tests.conftest import valid_yaml_config_file


def test_init_from_valid_dict(valid_config_dict):
    """Test that the Parameters class can be instantiated from a valid
    dictionary."""
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
    """Test that the Parameters class can be instantiated from a valid YAML
    config file."""
    p = Parameters.from_yaml(valid_yaml_config_file)
    q = Parameters.from_dict(valid_config_dict)

    assert p == q


def test_init_from_alternative_param_names(valid_config_dict):
    """Test that the Parameters class can be instantiated with alternative
    parameter names."""

    # create copy of valid_config_dict and rename some parameters
    valid_config_dict_alt_names = valid_config_dict.copy()
    valid_config_dict_alt_names['relb1'] = valid_config_dict_alt_names.pop('rel_b1')
    valid_config_dict_alt_names['b0_inhomogeneity'] = valid_config_dict_alt_names.pop('b0_inhom')

    p = Parameters.from_dict(valid_config_dict_alt_names)
    q = Parameters.from_dict(valid_config_dict)

    assert p == q


def test_export_to_yaml(valid_config_dict, empty_config_file):
    """Test that the Parameters class can be exported to a YAML config file."""

    # create Parameters object from valid_config_dict
    p = Parameters.from_dict(valid_config_dict)

    # export Parameters object to YAML config file
    p.to_yaml(empty_config_file)

    # create Parameters object from the exported config file
    q = Parameters.from_yaml(empty_config_file)

    # ensure that both objects are equal
    assert p == q


@pytest.mark.parametrize(
    'parameter, value',
    [
        ('t1', 3.0),
        ('t1', '5'),
        ('r1', 0.5),
        ('f', 0.5),
    ],
)
def test_update_water_pool_parameter(valid_config_dict, parameter, value):
    """Test that the water pool parameters are updated correctly."""
    p = Parameters.from_dict(valid_config_dict)

    # call p.update_water_pool() with the parameter and value
    getattr(p, 'update_water_pool')(**{parameter: value})

    # assert that the attribute was updated correctly
    assert getattr(p.water_pool, parameter) == float(value)
