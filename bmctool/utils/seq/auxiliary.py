"""
auxiliary.py
    Auxiliary functions for seq files.
"""
from pathlib import Path
from typing import Any, Union

import deprecation
import numpy as np
from pypulseq import Sequence
from pypulseq.Sequence.read_seq import __read_definitions as read_definitions
from pypulseq.Sequence.read_seq import __strip_line as strip_line

from bmctool.utils.seq.read import read_any_version


def get_definitions(seq_file: Union[str, Path] = None) \
        -> dict:
    """
    Reads all definitions directly from a seq-file.
    :param seq_file: sequence file to read the definitions from
    :return: dictionary with all definitions
    """
    with open(seq_file, 'r') as seq:
        while True:
            line = strip_line(seq)
            if line == -1:
                break
            elif line == '[DEFINITIONS]':
                dict_definitions = read_definitions(seq)
            else:
                pass

    return dict_definitions


def get_definition(seq_file: Union[str, Path] = None,
                   key: str = None) \
        -> Any:
    """
    Reads a single definition directly from a seq-file.
    :param seq_file: sequence file to read the definition from
    :param key: name of the dict entry of interest
    :return: dictionary entry for 'name'
    """
    dict_definitions = get_definitions(seq_file)
    if key in dict_definitions:
        value = dict_definitions[key]
    else:
        raise AttributeError(f'No definition called {key} in seq-file {seq_file}.')

    return value


@deprecation.deprecated(deprecated_in='0.3.2',
                        removed_in='1.0',
                        details="Use get_definition() function instead.")
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


def get_num_adc_events(seq_file: Union[str, Path]) -> int:
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
