import numpy as np
import pytest
from bmctool.simulation import BMCSim


def test_init_from_valid_params_and_seq_object(valid_parameters_object, valid_sequence_object):
    """Test BMCSim instantiation from valid parameters and sequence objects."""
    BMCSim(params=valid_parameters_object, seq=valid_sequence_object)


def test_init_from_valid_params_and_seq_file(valid_parameters_object, valid_seq_file):
    """Test BMCSim instantiation from valid parameters and sequence file."""
    BMCSim(params=valid_parameters_object, seq=valid_seq_file)


def test_init_from_seq_obj_and_file_match(valid_parameters_object, valid_sequence_object, valid_seq_file):
    """Test BMCSim instantiation from sequence object and file match."""
    sim1 = BMCSim(params=valid_parameters_object, seq=valid_sequence_object)
    sim2 = BMCSim(params=valid_parameters_object, seq=valid_seq_file)
    assert np.equal(sim1.offsets_ppm, sim2.offsets_ppm).all()
    assert np.equal(sim1.n_measure, sim2.n_measure).all()
    assert np.equal(sim1.m_init, sim2.m_init).all()


def test_verbose_flag_not_influence_result(valid_parameters_object, valid_sequence_object):
    """Test setting verbose flag."""
    sim1 = BMCSim(params=valid_parameters_object, seq=valid_sequence_object, verbose=True)
    sim2 = BMCSim(params=valid_parameters_object, seq=valid_sequence_object, verbose=False)
    sim1.run()
    sim2.run()

    assert np.equal(sim1.get_zspec(), sim2.get_zspec()).all()


def test_store_dynamics_option(valid_parameters_object, valid_sequence_with_single_gaussian_pulse):
    """Test setting store_dynamics option."""
    # ensure that nothing is stored when store_dynamics is set to 0
    sim = BMCSim(params=valid_parameters_object, seq=valid_sequence_with_single_gaussian_pulse, store_dynamics=0)
    sim.run()
    with pytest.raises(Warning, match='Dynamics were not stored.'):
        sim.get_dynamics()

    # ensure that values after each block are stored when store_dynamics is set to 1
    sim = BMCSim(params=valid_parameters_object, seq=valid_sequence_with_single_gaussian_pulse, store_dynamics=1)
    sim.run()
    times, values = sim.get_dynamics()
    assert len(times) == len(values)
    assert len(times) == 2  # 1 for initial state, 1 for RF block

    # ensure that values after each simulation step are stored when store_dynamics is set to 2
    sim = BMCSim(params=valid_parameters_object, seq=valid_sequence_with_single_gaussian_pulse, store_dynamics=2)
    sim.run()
    times, values = sim.get_dynamics()
    assert len(times) == len(values)
    assert len(times) == sim.params.options.max_pulse_samples + 1  # 1 for initial state, max_pulse_samples rf steps


def test_update_parameters(valid_parameters_object, valid_sequence_object):
    """Test updating parameters."""
    sim = BMCSim(params=valid_parameters_object, seq=valid_sequence_object)
    sim.update_params(valid_parameters_object)
    assert sim.params == valid_parameters_object
