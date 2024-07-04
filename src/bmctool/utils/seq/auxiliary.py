"""auxiliary.py Auxiliary functions for PyPulseq seq-file handling."""

from pathlib import Path

import numpy as np
from pypulseq.Sequence.read_seq import __read_definitions as read_definitions  # type: ignore
from pypulseq.Sequence.read_seq import __strip_line as strip_line


def get_definitions(filepath: str | Path) -> dict:
    """get_definitions Read all definitions directly from a seq-file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the seq-file

    Returns
    -------
    dict
        Dictionary of all definitions in the seq-file

    Raises
    ------
    AttributeError
        Raised if no definitions are found in the seq-file
    """
    with Path(filepath).open() as seq:
        while True:
            line = strip_line(seq)
            if line == -1:
                break
            elif line == '[DEFINITIONS]':
                dict_definitions: dict = read_definitions(seq)
            else:
                pass
    if not dict_definitions:
        raise AttributeError(f'No definitions found in file {filepath}.')
    return dict_definitions


def get_definition(filepath: str | Path, key: str) -> int | float | np.ndarray | None:
    """get_definition Read a single definition directly from a seq-file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the seq-file
    key : str, optional
        Name/key of the definition to be read

    Returns
    -------
    Any
        Value of the definition

    Raises
    ------
    AttributeError
        Raised if key is not found in the seq-file
    """
    dict_definitions: dict[str, int | float | np.ndarray | None] = get_definitions(filepath)
    if key in dict_definitions:
        return dict_definitions[key]

    raise AttributeError(f'No definition called {key} in file {filepath}.')


def get_num_adc_events(filepath: str | Path) -> int:
    """get_num_adc_events Reads number of ADC events in a sequence file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the seq-file

    Returns
    -------
    int
        Number of ADC events in the seq-file
    """
    adc_event_count = 0
    with Path(filepath).open() as seq:
        while True:
            line = strip_line(seq)
            if line == -1:
                break
            elif line == '[BLOCKS]':
                line = strip_line(seq)
                while line != '' and line != ' ' and line != '#':
                    block_event = np.fromstring(line, dtype=int, sep=' ')
                    if block_event[6] == 1:
                        adc_event_count += 1
                    line = strip_line(seq)
            else:
                pass
    return adc_event_count
