import re
from pathlib import Path

import numpy as np
from bmctool.utils.seq.write import insert_seq_file_header
from bmctool.utils.seq.write import round_number
from bmctool.utils.seq.write import write_seq
from bmctool.utils.seq.write import write_seq_defs


class _DummySeq:
    def __init__(self) -> None:
        self.defs: list[tuple[str, object]] = []
        self.written_to: str | None = None

    def set_definition(self, key: str, value: object) -> None:
        self.defs.append((key, value))

    def write(self, filename: str) -> None:
        self.written_to = filename
        Path(filename).write_text('# Created by pypulseq\n# dummy\n')


def test_round_number_rounds_to_significant_digits():
    assert round_number(1234.0, 2) == 1200.0
    assert round_number(0.012345, 2) == 0.012
    assert round_number(-0.012345, 2) == -0.012


def test_round_number_zero_returns_zero():
    assert round_number(0.0, 5) == 0.0


def test_insert_seq_file_header_inserts_lines_after_created_by(tmp_path):
    p = tmp_path / 'test.seq'
    p.write_text('# Created by pypulseq\n# Some other line\n')

    insert_seq_file_header(filepath=p, author='Alice')

    text = p.read_text().splitlines()
    assert text[0].startswith('# Created by')

    joined = '\n'.join(text)
    assert '# Created for Pulseq-CEST' in joined
    assert '# https://pulseq-cest.github.io/' in joined
    assert '# Created by: Alice' in joined
    assert re.search(r'# Created at: \d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2}', joined)


def test_write_seq_defs_translates_keys_and_rounds_scalars():
    seq = _DummySeq()

    seq_defs = {
        'b0': 3.0,
        'trec_m0': 1.234567891234,
        'custom': 7,
        'arr': np.array([1, 2, 3]),
    }

    write_seq_defs(seq=seq, seq_defs=seq_defs, use_matlab_names=True)

    keys = [k for k, _ in seq.defs]
    assert 'B0' in keys
    assert 'Trec_M0' in keys
    assert 'custom' in keys
    assert 'arr' in keys

    vals = dict(seq.defs)
    assert isinstance(vals['B0'], str)
    assert vals['B0'] == str(round_number(3.0, 9))
    assert isinstance(vals['Trec_M0'], str)
    assert vals['Trec_M0'] == str(round_number(1.234567891234, 9))
    assert isinstance(vals['custom'], str)
    assert vals['custom'] == str(round_number(7.0, 9))
    assert isinstance(vals['arr'], np.ndarray)


def test_write_seq_writes_file_and_inserts_header(tmp_path):
    seq = _DummySeq()
    out = tmp_path / 'out.seq'

    write_seq(seq=seq, seq_defs={'b0': 3.0}, filename=out, author='Bob', use_matlab_names=True)

    assert seq.written_to == str(out)
    text = out.read_text()
    assert '# Created for Pulseq-CEST' in text
    assert '# Created by: Bob' in text
