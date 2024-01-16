import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def calc_mtr_asym(
    m_z: np.ndarray,
    offsets: np.ndarray,
    n_interp: int = 1000,
) -> np.ndarray:
    """Calculate MTRasym from given magnetization vector.

    Parameter
    ---------
    m_z
        magnetization values
    offsets
        frequency offsets
    n_interp, optional
        Number of interpolation steps, by default 1000

    Return
    ------
    np.ndarray
        Array containing the MTRasym values
    """

    x_interp = np.linspace(np.min(offsets), np.max(np.absolute(offsets)), n_interp)
    y_interp = np.interp(x_interp, offsets, m_z)
    asym = y_interp[::-1] - y_interp
    return np.interp(offsets, x_interp, asym)


def normalize_data(
    m_z: np.ndarray,
    offsets: np.ndarray,
    threshold: int | float | list | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize magnetization values.

    Parameter
    ---------
    m_z
        not normalized magnetization values
    offsets
        frequency offsets
    threshold
        threshold for data splitting. If single value, abs(val) is used, else min/max values.

    Return
    ------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the normalized magnetization vector and the corresponding offsets.
    """

    offsets, data, norm = split_data(m_z, offsets, threshold)

    if norm is not None:
        m_z = np.divide(data, np.mean(norm), out=np.zeros_like(data), where=np.mean(norm) != 0)

    return m_z, offsets


def split_data(
    m_z: np.ndarray,
    offsets: np.ndarray,
    threshold: int | float | list | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Split magnetization vector into data and normalization data.

    Parameter
    ---------
    m_z
        magnetization values
    offsets
        frequency offsets
    threshold
        threshold for data splitting. If single value, abs(val) is used, else min/max values.

    Return
    ------
    Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]
        Tuple containing offsets, data and normalization data.

    Raises
    ------
    TypeError
        If threshold is not of type int, float, list or np.ndarray.
    """

    if isinstance(threshold, int | float):
        th_high = np.abs(threshold)
        th_low = -th_high
    elif isinstance(threshold, list) and len(threshold) == 2:
        th_high = max(threshold)
        th_low = min(threshold)
    elif isinstance(threshold, np.ndarray) and threshold.size == 2:
        th_high = np.max(threshold)
        th_low = np.min(threshold)
    else:
        raise TypeError(f"Threshold of type '{type(threshold)}' not supported.")

    idx_data = np.where(np.logical_and(offsets > th_low, offsets < th_high))
    idx_norm = np.where(np.logical_or(offsets <= th_low, offsets >= th_high))

    if idx_norm[0].size == 0:
        return offsets, m_z, None

    offsets = offsets[idx_data]
    data = m_z[idx_data]
    norm = m_z[idx_norm]

    return offsets, data, norm


def plot_z(
    m_z: np.ndarray,
    offsets: np.ndarray | None = None,
    normalize: bool = False,
    norm_threshold: int | float | list | np.ndarray = 295,
    invert_ax: bool = True,
    plot_mtr_asym: bool = False,
    title: str = 'spectrum',
    x_label: str = 'offsets [ppm]',
    y_label: str = 'signal',
) -> Figure:
    """plot_z Plot Z-spectrum according to the given parameters.

    Parameters
    ----------
    m_z : np.array
        Magnetization values.
    offsets : np.array, optional
        Offset values, by default None
    normalize : bool, optional
        Flag to activate normalization of magnetization values, by default False
    norm_threshold : Union[int, float, list, np.ndarray], optional
        Threshold used for normalization, by default 295
    invert_ax : bool, optional
        Flag to activate the inversion of the x-axis, by default True
    plot_mtr_asym : bool, optional
        Flag to activate the plotting of MTRasym values, by default False
    title : str, optional
        Figure title, by default "spectrum"
    x_label : str, optional
        Label of x-axis, by default "offsets [ppm]"
    y_label : str, optional
        Label of y-axis, by default "signal"

    Returns
    -------
    Figure
        Matplotlib figure object.
    """

    if offsets is None:
        offsets = np.arange(len(m_z))

    if normalize:
        m_z, offsets = normalize_data(m_z, offsets, norm_threshold)

    fig, ax1 = plt.subplots()
    ax1.set_ylim([round(min(m_z) - 0.05, 2), round(max(m_z) + 0.05, 2)])
    ax1.set_ylabel(y_label, color='b')
    ax1.set_xlabel(x_label)
    plt.plot(offsets, m_z, '.--', label='$Z$', color='b')
    if invert_ax:
        plt.gca().invert_xaxis()
    ax1.tick_params(axis='y', labelcolor='b')

    if plot_mtr_asym:
        mtr_asym = calc_mtr_asym(m_z=m_z, offsets=offsets)

        ax2 = ax1.twinx()
        ax2.set_ylim([round(min(mtr_asym) - 0.01, 2), round(max(mtr_asym) + 0.01, 2)])
        ax2.set_ylabel('$MTR_{asym}$', color='y')
        ax2.plot(offsets, mtr_asym, label='$MTR_{asym}$', color='y')
        ax2.tick_params(axis='y', labelcolor='y')
        fig.tight_layout()

    plt.title(title)
    plt.show()

    return fig
