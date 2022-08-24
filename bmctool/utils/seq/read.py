"""
read.py
    Auxiliary functions for reading of seq files and seq file entries.
"""
import deprecation
from typing import Union
from pathlib import Path
from pypulseq.Sequence.sequence import Sequence


@deprecation.deprecated(deprecated_in='0.4.1',
                        removed_in='1.0',
                        details="Function not needed for pypulseq >= 1.3.1")
def get_minor_version(seq_file: Union[str, Path]) -> int:
    """
    Reads minor version from a seq file.
    :param seq_file: path to the sequence file to read into the Sequence object
    :return version: version from the sequence file
    """
    with open(seq_file) as file:
        for line in file:
            if line.startswith('minor'):
                return int(line[len('minor '):])


@deprecation.deprecated(deprecated_in='0.4.1',
                        removed_in='1.0',
                        details="Function not needed for pypulseq >= 1.3.1")
def read_any_version(seq_file: Union[str, Path],
                     seq: Sequence = None) \
        -> Sequence:
    """
    Reads a sequence file (seq_file) independent of the (py)pulseq version.
    :param seq_file: path to the sequence file to read into the Sequence object
    :param seq: the sequence to read the seq file into. If not provided, a new Sequence object is instantiated
    :return seq: Sequence object
    """
    version = get_minor_version(seq_file)
    if not seq:
        seq = Sequence()
    if version in [2, 3]:
        seq.read(seq_file)
    else:
        raise ValueError('Version', version, 'can not be converted.')
    return seq
