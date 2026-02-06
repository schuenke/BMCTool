import numpy as np
import pypulseq as pp  # type: ignore
from bmctool.utils.pulses.make_hsexp import calculate_frequency
from bmctool.utils.pulses.make_hsexp import calculate_window_modulation
from bmctool.utils.pulses.make_hsexp import generate_hsexp_dict
from bmctool.utils.pulses.make_hsexp import make_hsexp


def test_calculate_window_modulation_is_zero_at_start_and_one_at_end():
    t0 = 1.0
    t = np.array([0.0, t0])

    w = calculate_window_modulation(t=t, t0=t0)

    assert w.shape == (2,)
    assert np.isclose(w[0], 0.0)
    assert np.isclose(w[1], 1.0)


def test_calculate_frequency_freq_factor_flips_sign():
    t0 = 1.0
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    bandwidth = 2500.0
    ef = 3.5

    f_pos = calculate_frequency(t=t, t0=t0, bandwidth=bandwidth, ef=ef, freq_factor=1)
    f_neg = calculate_frequency(t=t, t0=t0, bandwidth=bandwidth, ef=ef, freq_factor=-1)

    assert np.allclose(f_pos, -f_neg)
    assert np.all(f_pos <= 0.0)
    assert np.all(f_neg >= 0.0)


def test_make_hsexp_tip_down_windows_start_of_pulse():
    system = pp.Opts()

    rf = make_hsexp(
        amp=1.0,
        t_p=200e-6,
        mu=20.0,
        bandwidth=2000.0,
        t_window=50e-6,
        ef=3.0,
        tip_down=True,
        pos_offset=True,
        system=system,
    )

    mag = np.abs(rf.signal)
    mid = mag.size // 2

    assert np.iscomplexobj(rf.signal)
    assert mag[0] < mag[mid]


def test_make_hsexp_tip_up_windows_end_of_pulse():
    system = pp.Opts()

    rf = make_hsexp(
        amp=1.0,
        t_p=200e-6,
        mu=20.0,
        bandwidth=2000.0,
        t_window=50e-6,
        ef=3.0,
        tip_down=False,
        pos_offset=True,
        system=system,
    )

    mag = np.abs(rf.signal)
    mid = mag.size // 2

    assert np.iscomplexobj(rf.signal)
    assert mag[-1] < mag[mid]


def test_generate_hsexp_dict_returns_all_variants_and_they_differ():
    system = pp.Opts()

    d = generate_hsexp_dict(
        amp=1.0,
        t_p=200e-6,
        mu=20.0,
        bandwidth=2000.0,
        t_window=50e-6,
        ef=3.0,
        system=system,
    )

    assert set(d.keys()) == {'pre_pos', 'pre_neg', 'post_pos', 'post_neg'}

    pre_pos = d['pre_pos']
    pre_neg = d['pre_neg']

    assert pre_pos.signal.shape == pre_neg.signal.shape
    assert not np.allclose(pre_pos.signal, pre_neg.signal)
