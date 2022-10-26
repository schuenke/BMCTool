"""
write.py
    Auxiliary functions for writing seq files.
"""
import math
from datetime import datetime
from os import fdopen, remove
from pathlib import Path
from shutil import move, copymode
from tempfile import mkstemp
from typing import Union

import numpy as np
from pypulseq.Sequence.sequence import Sequence

from bmctool.utils.seq.conversion import convert_seq_12_to_13


def round_number(number, significant_digits):
    """
    Rounds a number to the specified number of significant digits
    :param number: number to be rounded
    :param significant_digits: number of significant digits
    :return:
    """
    if number != 0:
        return round(number, significant_digits - int(math.floor(math.log10(abs(number)))) - 1)
    else:
        return 0


def insert_seq_file_header(filepath: Union[str, Path],
                           author: str):
    """
    Inserts header information into seq-file
    :param filepath: path to the sequence file
    :param author: name of the author of the file
    """
    # create a temp file
    tmp, abs_path = mkstemp()

    in_position = False
    with fdopen(tmp, 'w') as new_file:
        with open(filepath) as old_file:
            for line in old_file:
                if line.startswith('# Created by'):
                    new_file.write(line)
                    in_position = True
                else:
                    if in_position:
                        new_file.write(f"\n")
                        new_file.write(f"# Created for Pulseq-CEST\n")
                        new_file.write(f"# https://pulseq-cest.github.io/\n")
                        new_file.write(f"# Created by: {author}\n")
                        new_file.write(f"# Created at: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}\n")
                        new_file.write(f"\n")
                        in_position = False
                    else:
                        new_file.write(line)

    # copy permissions from old file to new file
    copymode(filepath, abs_path)
    # remove old file
    remove(filepath)
    # move new file
    move(abs_path, filepath)


def write_seq_defs(seq: Sequence,
                   seq_defs: dict,
                   use_matlab_names: bool) \
        -> Sequence:
    """
    Writes seq-file 'Definitions' from dictionary
    :param seq: pypulseq Sequence object
    :param seq_defs: dictionary with all entries that should be written into 'Definitions' of the seq-file
    :param use_matlab_names: set to True to use the same variable names as in Matlab
    :return:
    """
    if use_matlab_names:
        # define MATLAB names
        translator = {'b0': 'B0',
                      'b1cwpe': 'B1cwpe',
                      'dcsat': 'DCsat',
                      'm0_offset': 'M0_offset',
                      'n_slices': 'nSlices',
                      'ti': 'TI',
                      'trec': 'Trec',
                      'trec_m0': 'Trec_M0',
                      'tsat': 'Tsat'
                      }

        # create new dict with correct names (needs to be done before to be able to sort it correctly)
        dict_ = {}
        for k, v in seq_defs.items():
            # convert names
            if k in translator:
                k = translator[k]

            # write entry
            dict_.update({k: v})
    else:
        dict_ = seq_defs

    # write definitions in alphabetical order and convert to correct value types
    for k, v in sorted(dict_.items()):
        # convert value types
        if type(v) == np.ndarray:
            pass
        elif type(v) in [int, float, np.float32, np.float64, np.float]:
            v = str(round_number(v, 6))
        else:
            v = str(v)
        seq.set_definition(key=k, value=v)

    return seq


def write_seq(seq: Sequence,
              seq_defs: dict,
              filename: Union[str, Path],
              author: str,
              use_matlab_names: bool = True,
              convert_to_1_3: bool = False):
    """
    Writes the seq-file according to given arguments
    :param seq: pypulseq Sequence object
    :param seq_defs: dictionary with all entries that should be written into 'Definitions' of the seq-file
    :param filename: Path or string with the filename
    :param author: name of the author of the file
    :param use_matlab_names: set to True to use the same variable names as in Matlab
    :param convert_to_1_3: set to True to convert a version 1.2 seq-file to a pseudo version 1.3 file
    """
    # write definitions
    write_seq_defs(seq, seq_defs, use_matlab_names)

    # write *.seq file
    seq.write(filename)

    # insert header
    insert_seq_file_header(filepath=filename, author=author)

    # convert to pypulseq version 1.3
    if convert_to_1_3:
        convert_seq_12_to_13(filename)
