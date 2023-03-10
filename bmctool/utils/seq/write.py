"""
write.py
    Auxiliary functions for writing seq files.
"""
import math
from datetime import datetime
from os import fdopen, remove
from pathlib import Path
from shutil import copymode, move
from tempfile import mkstemp
from typing import Union

import numpy as np
from pypulseq.Sequence.sequence import Sequence


def round_number(number: float, significant_digits: int) -> float:
    """
    round_number Rounds a number to the specified number of significant digits

    Parameters
    ----------
    number : float
        number to be rounded
    significant_digits : int
        number of significant digits

    Returns
    -------
    float
        rounded number
    """
    if number != 0:
        return round(number, significant_digits - int(math.floor(math.log10(abs(number)))) - 1)
    return 0.0


def insert_seq_file_header(filepath: Union[str, Path], author: str) -> None:
    """
    insert_seq_file_header Inserts header information into seq-file

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the seq-file
    author : str
        author name
    """
    # create a temp file
    tmp, abs_path = mkstemp()

    in_position = False
    with fdopen(tmp, "w") as new_file:
        with open(filepath, "r") as old_file:
            for line in old_file:
                if line.startswith("# Created by"):
                    new_file.write(line)
                    in_position = True
                else:
                    if in_position:
                        new_file.write("\n")
                        new_file.write("# Created for Pulseq-CEST\n")
                        new_file.write("# https://pulseq-cest.github.io/\n")
                        new_file.write(f"# Created by: {author}\n")
                        new_file.write(f"# Created at: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}\n")
                        new_file.write("\n")
                        in_position = False
                    else:
                        new_file.write(line)

    # copy permissions from old file to new file
    copymode(filepath, abs_path)
    # remove old file
    remove(filepath)
    # move new file
    move(abs_path, filepath)


def write_seq_defs(seq: Sequence, seq_defs: dict, use_matlab_names: bool) -> Sequence:
    """
    write_seq_defs Writes seq-file 'Definitions' from dictionary

    Parameters
    ----------
    seq : Sequence
        Pulseq sequence object
    seq_defs : dict
        dictionary with sequence definitions
    use_matlab_names : bool
        flag to use MATLAB names for sequence definitions

    Returns
    -------
    Sequence
        _description_
    """
    if use_matlab_names:
        translator = {
            "b0": "B0",
            "b1cwpe": "B1cwpe",
            "b1pa": "B1pa",
            "b1rms": "B1rms",
            "dcsat": "DCsat",
            "freq": "FREQ",
            "m0_offset": "M0_offset",
            "n_slices": "nSlices",
            "ti": "TI",
            "trec": "Trec",
            "trec_m0": "Trec_M0",
            "tsat": "Tsat",
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
        if type(v) == np.ndarray:
            pass
        elif type(v) in [int, float, np.float32, np.float64, np.float]:
            v = str(round_number(v, 9))
        else:
            v = str(v)
        try:
            seq.set_definition(key=k, value=v)
        except TypeError:
            seq.set_definition(key=k, val=v)

    return seq


def write_seq(
    seq: Sequence,
    seq_defs: dict,
    filename: Union[str, Path],
    author: str,
    use_matlab_names: bool = True,
) -> None:
    """
    write_seq Writes the seq-file according to given arguments

    Parameters
    ----------
    seq : Sequence
        PyPulseq sequence object
    seq_defs : dict
        dictionary with sequence definitions
    filename : Union[str, Path]
        name of the seq-file
    author : str
        author name
    use_matlab_names : bool, optional
        flag to use MATLAB names for sequence definitions, by default True
    """
    # write definitions
    write_seq_defs(seq, seq_defs, use_matlab_names)

    # write *.seq file
    seq.write(str(filename))

    # insert header
    insert_seq_file_header(filepath=filename, author=author)
