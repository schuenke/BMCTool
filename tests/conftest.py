from pathlib import Path

import numpy as np
import pypulseq as pp  # type: ignore
import pytest
import yaml
from bmctool.parameters import CESTPool
from bmctool.parameters import MTPool
from bmctool.parameters import Options
from bmctool.parameters import Parameters
from bmctool.parameters import System
from bmctool.parameters import WaterPool


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


@pytest.fixture(scope='session')
def valid_water_pool_object():
    return WaterPool(r1=1.0, r2=2.0)


@pytest.fixture(scope='session')
def valid_cest_pool_object():
    return CESTPool(f=0.001, r1=1.0, r2=60, k=100, dw=3.0)


@pytest.fixture(scope='session')
def valid_mt_pool_object():
    return MTPool(r1=1.0, r2=2.0, k=3.0, f=0.5, dw=5.0, lineshape='lorentzian')


@pytest.fixture(scope='session')
def valid_system_object():
    return System(b0=3.0, gamma=42.5764, b0_inhom=0.0, rel_b1=1.0)


@pytest.fixture(scope='session')
def valid_options_object():
    return Options()


@pytest.fixture(scope='session')
def valid_parameters_object(
    valid_water_pool_object, valid_cest_pool_object, valid_mt_pool_object, valid_system_object, valid_options_object
):
    return Parameters(
        water_pool=valid_water_pool_object,
        cest_pools=[valid_cest_pool_object],
        mt_pool=valid_mt_pool_object,
        system=valid_system_object,
        options=valid_options_object,
    )


@pytest.fixture(scope='session')
def valid_sequence_object():
    seq = pp.Sequence()
    sys = pp.Opts()

    offsets_ppm = np.linspace(-5, 5, 11)
    offsets_hz = offsets_ppm * 42.5764 * 3
    delay = pp.make_delay(100e-3)
    pseudo_adc = pp.make_adc(num_samples=1, duration=1e-3)

    for offset in offsets_hz:
        rf = pp.make_gauss_pulse(flip_angle=10 * np.pi, duration=5e-3, freq_offset=offset, system=sys)
        seq.add_block(rf)
        seq.add_block(delay)
        seq.add_block(pseudo_adc)

    seq.set_definition('offsets_ppm', offsets_ppm)

    return seq


@pytest.fixture(scope='session')
def valid_sequence_with_single_gaussian_pulse():
    seq = pp.Sequence()
    sys = pp.Opts()

    seq.set_definition('offsets_ppm', [0])

    rf = pp.make_gauss_pulse(flip_angle=np.pi / 2, duration=5e-3, freq_offset=0, system=sys)
    pseudo_adc = pp.make_adc(num_samples=1, duration=1e-3)

    seq.add_block(rf)
    seq.add_block(pseudo_adc)

    return seq


@pytest.fixture(scope='session')
def valid_seq_file(tmp_path_factory, valid_sequence_object):
    fn = tmp_path_factory.mktemp('valid_seq_file') / 'valid_seq_file.seq'
    valid_sequence_object.write(fn)
    return fn
