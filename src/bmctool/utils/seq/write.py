"""Auxiliary functions for writing seq files.

These function are used in the pulseq-cest library to write seq files matching the MATLAB style.
The pulseq-cest library can be found at: https://github.com/kherz/pulseq-cest-library
"""

import math
from datetime import datetime
from os import fdopen
from pathlib import Path
from shutil import move
from tempfile import mkstemp

import numpy as np
from pypulseq import Sequence  # type: ignore


def round_number(number: float, significant_digits: int) -> float:
    """Round a number to the specified number of significant digits.

    Parameter
    ---------
    number
        number to be rounded
    significant_digits
        number of significant digits

    Return
    ------
    float
        rounded number
    """
    if number != 0:
        return round(number, significant_digits - int(math.floor(math.log10(abs(number)))) - 1)
    return 0.0


def insert_seq_file_header(filepath: str | Path, author: str) -> None:
    """Insert header information into seq-file.

    Parameter
    ---------
    filepath
        Path to pypulseq sequence file
    author
        name of the author
    """
    # create a temp file
    tmp, abs_path = mkstemp()

    in_position = False
    with fdopen(tmp, 'w') as new_file, Path(filepath).open() as old_file:
        for line in old_file:
            if line.startswith('# Created by'):
                new_file.write(line)
                in_position = True
            else:
                if in_position:
                    new_file.write('\n')
                    new_file.write('# Created for Pulseq-CEST\n')
                    new_file.write('# https://pulseq-cest.github.io/\n')
                    new_file.write(f'# Created by: {author}\n')
                    new_file.write(f"# Created at: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}\n")
                    new_file.write('\n')
                    in_position = False
                else:
                    new_file.write(line)

    # copy permissions from old file to new file
    # remove old file
    Path(filepath).unlink()
    # move new file
    move(abs_path, filepath)


def write_seq_defs(seq: Sequence, seq_defs: dict, use_matlab_names: bool) -> Sequence:
    """Write seq-file 'Definitions' from dictionary.

    Parameter
    ---------
    seq
        PyPulseq sequence object
    seq_defs
        dictionary with sequence definitions
    use_matlab_names
        flag to use MATLAB names for sequence definitions

    Return
    ------
    seq
        PyPulseq sequence object
    """
    if use_matlab_names:
        translator = {
            'b0': 'B0',
            'b1cwpe': 'B1cwpe',
            'b1pa': 'B1pa',
            'b1rms': 'B1rms',
            'dcsat': 'DCsat',
            'freq': 'FREQ',
            'm0_offset': 'M0_offset',
            'n_slices': 'nSlices',
            'ti': 'TI',
            'trec': 'Trec',
            'trec_m0': 'Trec_M0',
            'tsat': 'Tsat',
        }

        # create new dict with correct names and values
        dict_ = {}
        for k, v in seq_defs.items():
            k = translator.get(k, k)

            # write entry
            dict_.update({k: v})
    else:
        dict_ = seq_defs

    # write definitions in alphabetical order and convert to correct value types
    for k, v in sorted(dict_.items()):
        # convert value types
        if isinstance(v, np.ndarray):
            pass
        elif isinstance(v, int | float | np.float32 | np.float64):
            v = str(round_number(float(v), 9))

        seq.set_definition(key=k, value=v)

    return seq


def write_seq(
    seq: Sequence,
    seq_defs: dict,
    filename: str | Path,
    author: str,
    use_matlab_names: bool = True,
) -> None:
    """Write seq-file according to given arguments.

    Parameter
    ---------
    seq
        PyPulseq sequence object
    seq_defs
        dictionary with sequence definitions
    filename
        name of the seq-file
    author
        name of the author
    use_matlab_names, optional
        flag to use MATLAB names for sequence definitions, by default True
    """
    # write definitions
    write_seq_defs(seq, seq_defs, use_matlab_names)

    # write *.seq file
    seq.write(str(filename))

    # insert header
    insert_seq_file_header(filepath=filename, author=author)
