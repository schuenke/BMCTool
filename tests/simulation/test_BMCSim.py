import numpy as np
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


def test_verbose_flag(valid_parameters_object, valid_sequence_object):
    """Test setting verbose flag."""
    sim1 = BMCSim(params=valid_parameters_object, seq=valid_sequence_object, verbose=True)
    sim2 = BMCSim(params=valid_parameters_object, seq=valid_sequence_object, verbose=False)
    sim1.run()
    sim2.run()

    assert np.equal(sim1.get_zspec(), sim2.get_zspec()).all()


def test_update_parameters(valid_parameters_object, valid_sequence_object):
    """Test updating parameters."""
    sim = BMCSim(params=valid_parameters_object, seq=valid_sequence_object)
    sim.update_params(valid_parameters_object)
    assert sim.params == valid_parameters_object
