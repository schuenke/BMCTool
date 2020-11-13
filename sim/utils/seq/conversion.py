"""
conversion.py
    Functions to convert between different versions of seq files.
"""
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove


def convert_seq_12_to_pseudo_13(file_path: str):
    """
    Converts version 1.2 seq-files to pseudo version 1.3 seq-files.
    :param file_path: path to the sequence file that should be converted
    """

    # create a temp file
    tmp, abs_path = mkstemp()

    in_blocks = False
    with fdopen(tmp, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if line.startswith('minor'):
                    if int(line[len('minor '):]) == 2:
                        new_file.write(line.replace('2', '3'))
                    else:
                        raise Exception(f'Version of seq-file (v. 1.{int(line[len("minor "):])}) '
                                        f'differs from expected version 1.2. Conversion aborted!')
                elif all(x in line for x in ['RF', 'GX', 'GY', 'GZ', 'ADC']):
                    new_file.write(''.join([line.strip(), ' EXT\n']))
                elif line.startswith('[BLOCKS]'):
                    new_file.write(line)
                    in_blocks = True
                else:
                    if in_blocks and line.strip() != '' and len(line.strip().split()) == 7:
                        block_list = line.strip().split()
                        block_list.append('0')  # add pseudo EXT entry
                        block_list.append('\n')  # append line ending
                        new_file.write(' '.join([f'{x:>3}' for x in block_list]))
                    else:
                        new_file.write(line)
                        in_blocks = False

    # copy permissions from old file to new file
    copymode(file_path, abs_path)
    # remove old file
    remove(file_path)
    # move new file
    move(abs_path, file_path)


def convert_seq_13_to_12(file_path: str,
                         temp: bool = False) \
        -> str:
    """
    Converts (pseudo) version 1.3 seq-files to version 1.2 seq-files.
    :param file_path: path to the sequence file that should be converted
    :param temp: toggle temporary conversion. Default False: the file is converted in place. If True: a temporary
                converted file is written and its path is returned
    :return path: if temp=True, this function returns the path to the converted file. The deletion needs to be handled
                independently after usage
    """

    # create a temp file
    tmp, abs_path = mkstemp(suffix='_temp')
    in_blocks = False

    with fdopen(tmp, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if line.startswith('minor'):
                    if int(line[len('minor '):]) == 3:
                        new_file.write(line.replace('3', '2'))
                    else:
                        raise Exception(f'Version of seq-file (v. 1.{int(line[len("minor "):])}) '
                                        f'differs from expected version 1.3. Conversion aborted!')
                elif all(x in line for x in ['RF', 'GX', 'GY', 'GZ', 'ADC', 'EXT']):
                    new_file.write(line.replace('EXT', ''))
                elif line.startswith('[BLOCKS]'):
                    new_file.write(line)
                    in_blocks = True
                else:
                    if in_blocks and line.strip() != '' and len(line.strip().split()) == 8:
                        block_list = line.strip().split()[:-1]  # remove last entry
                        block_list.append('\n')  # append line ending
                        new_file.write(' '.join([f'{x:>3}' for x in block_list]))
                    else:
                        new_file.write(line)
                        in_blocks = False

    if temp:
        return abs_path
    else:
        # copy permissions from old file to new file
        copymode(file_path, abs_path)
        # remove old file
        remove(file_path)
        # move new file
        move(abs_path, file_path)
