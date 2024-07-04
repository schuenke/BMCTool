from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope='session')
def valid_config_dict():
    return {
        'water_pool': {
            'r1': 1.0,
            'r2': 2.0,
            'f': 1.0,
        },
        'cest_pool': {
            'cest_1': {'f': 0.001, 't1': 1.0, 't2': 0.1, 'k': 100, 'dw': 1.0},
            'cest_2': {'f': 0.002, 't1': 2.0, 't2': 0.2, 'k': 2000, 'dw': -2.0},
        },
        'verbose': True,
        'reset_init_mag': True,
        'scale': 1.0,
        'max_pulse_samples': 200,
        'b0': 3.0,
        'gamma': 42.5764,
        'b0_inhom': 0.0,
        'rel_b1': 1.0,
    }


@pytest.fixture(scope='session')
def valid_config_dict_only_water():
    return {
        'water_pool': {
            'r1': 1.0,
            'r2': 2.0,
            'f': 1.0,
        },
        'verbose': True,
        'reset_init_mag': True,
        'scale': 1.0,
        'max_pulse_samples': 200,
        'b0': 3.0,
        'gamma': 42.5764,
        'b0_inhom': 0.0,
        'rel_b1': 1.0,
    }


@pytest.fixture(scope='session')
def valid_yaml_config_file(valid_config_dict, tmp_path_factory):
    fn = tmp_path_factory.mktemp('valid_config_file') / 'valid_config_file.yaml'
    with Path(fn).open('w') as f:
        yaml.dump(valid_config_dict, f)
    f.close()
    return fn


@pytest.fixture(scope='session')
def empty_config_file(tmp_path_factory):
    """Create an empty config file."""
    fn = tmp_path_factory.mktemp('empty_config') / 'empty_config.yaml'
    with Path(fn).open('w') as f:
        f.write('')
    f.close()
    return fn
