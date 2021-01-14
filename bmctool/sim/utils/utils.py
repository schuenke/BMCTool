"""
utils.py
    Useful additional functions
"""
import numpy as np
from pypulseq.Sequence.sequence import Sequence


def get_noise_params(mean: float = None,
                     std: float = None):
    """
    function to define standard noise parameters of mean=0 and std=0.005 for the missing parameters
    :param mean: the mean value of the gaussian function or None
    :param std: the standard deviation of the gaussian function or None
    :return (mean, std): previously defined values or the standard values
    """
    if not mean:
        mean = 0
    if not std:
        mean = 0.005
    return mean, std


def sim_noise(data: (float, np.ndarray, dict),
              mean: float = 0,
              std: float = 0.005,
              set_vals: (bool, tuple) = True):
    """
    simulating gaussian noise onto data
    :param data: the desired data to simulate noise on
    :param mean: the mean value of the gaussian function
    :param std: the standard deviation of the gaussian function
    :param set_vals: tuple to set (mean, std) or bool to set standard noise params of mean=0 and std=0.005
    :return output: data with added noise
    """
    if type(set_vals) is tuple:
        mean = set_vals[0]
        std = set_vals[1]
    elif set_vals:
        mean, std = get_noise_params(mean, std)
    ret_float = False
    if type(data) is dict:
        output = {}
        output.update({k: sim_noise(data=v, mean=mean, std=std) for k, v in data.items()})
    elif type(data) is list:
        output = [sim_noise(v) for v in data]
    else:
        if type(data) is float:
            data = np.array(data)
            ret_float = True
        elif type(data) is np.ndarray:
            data = data
        else:
            raise ValueError('Can only simulate noise on floats, arrays or lists/ dicts containing floats or arrays')
        noise = np.random.normal(mean, std, data.shape)
        output = data + noise
    if ret_float:
        output = float(output)
    return output


def get_offsets(seq: Sequence = None,
                seq_file: str = None) \
        -> list:
    """
    read the offsets either from the sequence file or from the Sequence object
    :param seq_file: sequence file to read the offsets from
    :param seq: Sequence object to get the offsets from
    :return: list of offsets
    """
    if not seq and not seq_file:
        raise ValueError('You need to pass either the sequence filename or the Sequence object to get offsets.')
    if not seq:
        seq = Sequence(version=1.3)
        seq.read(seq_file)
    try:
        offsets = list(seq.definitions['offsets_ppm'])
    except ValueError:
        print('Could not read offsets from seq-file.')
    return offsets


def check_m0_scan(seq: Sequence = None,
                  seq_file: str = None) \
        -> bool:
    """
    check wether m0 simulation is defined in either the sequence file or the Sequence object
    :param seq_file: sequence file to read the offsets from
    :param seq: Sequence object to get the offsets from
    :return: boolean
    """
    if not seq and not seq_file:
        raise ValueError('You need to pass either the sequence filename or the Sequence object to get m0_scan.')
    if not seq:
        seq = Sequence(version=1.3)
        seq.read(seq_file)
    if 1 in seq.definitions['run_m0_scan'] or 'True' in seq.definitions['run_m0_scan']:
        return True
    else:
        return False


def get_num_adc_events(seq: Sequence = None,
                       seq_file: str = None) \
        -> int:
    """
    Reads number of ADC events (should equal number of offsets)
    :param seq: Sequence object to get the offsets from
    :param seq_file: sequence file to read the offsets from
    :return: num_adc_events
    """
    if not seq and not seq_file:
        raise ValueError('You need to pass either the sequence filename or the Sequence object to get offsets for the '
                         'ADC events.')
    offsets = get_offsets(seq_file)
    num_adc_events = len(offsets)
    return num_adc_events
