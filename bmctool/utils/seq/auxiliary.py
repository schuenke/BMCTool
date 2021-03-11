"""
auxiliary.py
    Auxiliary functions for seq files.
"""
from pypulseq.Sequence.sequence import Sequence
from bmctool.utils.seq.read import read_any_version


def get_offsets(seq: Sequence = None,
                seq_file: str = None) \
        -> list:
    """
    Reads offsets either from a seq file or from a Sequence object.
    :param seq_file: sequence file to read the offsets from
    :param seq: Sequence object to get the offsets from
    :return: list of offsets
    """
    if not seq and not seq_file:
        raise ValueError('You need to pass either the sequence filename or the Sequence object.')
    if not seq:
        seq = read_any_version(seq_file=seq_file)
    offsets = seq.definitions['offsets_ppm']
    return offsets


def get_num_adc_events(seq: Sequence = None,
                       seq_file: str = None) \
        -> int:
    """
    Reads number of ADC events (should equal number of offsets).
    :param seq: Sequence object to get the offsets from
    :param seq_file: sequence file to read the offsets from
    :return: num_adc_events
    """
    if not seq and not seq_file:
        raise ValueError('You need to pass either the sequence filename or the Sequence object.')
    offsets = get_offsets(seq_file=seq_file)
    num_adc_events = len(offsets)
    return num_adc_events