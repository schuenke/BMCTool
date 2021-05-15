"""
auxiliary.py
    Auxiliary functions for seq files.
"""
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.Sequence.read_seq import __strip_line as strip_line
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
    offsets = seq.dict_definitions['offsets_ppm']
    return offsets


def get_num_adc_events(seq_file: str) -> int:
    """
    Reads number of ADC events in a sequence file
    :param seq_file: sequence file to read the num_adc_events from
    :return: num_adc_events
    """
    adc_event_count = 0
    with open(seq_file, 'r') as seq:
        while True:
            line = strip_line(seq)
            if line == -1:
                break
            elif line == '[BLOCKS]':
                line = strip_line(seq)
                while line != '' and line != ' ' and line != '#':
                    block_event = np.fromstring(line, dtype=int, sep=' ')
                    if block_event[6] == 1:
                        adc_event_count += 1  # count number of events before 1st adc
                    line = strip_line(seq)
            else:
                pass
    return adc_event_count
