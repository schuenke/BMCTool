"""
eval.py
    Tool independent functions for plotting and calculations
"""
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def calc_mtr_asym(z: np.ndarray,
                  offsets: np.ndarray,
                  n_interp: int = 1000) \
        -> np.ndarray:
    """
    Calculates MTRasym from the magnetization vector.
    :param z: magnetization values
    :param offsets: offset values
    :param n_interp: number of points used for interpolation
    :return: MTRasym
    """
    x_interp = np.linspace(np.min(offsets),
                           np.max(np.absolute(offsets)),
                           n_interp)
    y_interp = np.interp(x_interp, offsets, z)
    asym = y_interp[::-1] - y_interp
    return np.interp(offsets, x_interp, asym)


def normalize_data(mz: np.ndarray,
                   offsets: np.ndarray,
                   threshold: Union[int, float, list, np.ndarray],
                   **kwargs) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes given data by the mean of values corresponding to offsets exceeding the given threshold.
    :param mz: y-values
    :param offsets: x-values (e.g. offsets, inversion times)
    :param threshold: threshold for data splitting
    return: tuple containing normalized data and corresponding offsets
    """
    offsets, data, norm = split_data(mz, offsets, threshold)

    if norm is not None:
        mz = np.divide(data, np.mean(norm), out=np.zeros_like(data), where=np.mean(norm) != 0)

    return mz, offsets


def split_data(mz: np.ndarray,
               offsets: np.ndarray,
               threshold: Union[int, float, list, np.ndarray]) \
        -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """
    Splits data into data and normalization offsets depending on the given threshold
    :param mz: y-values
    :param offsets: x-values (e.g. offsets, inversion times)
    :param threshold: threshold for data splitting
    """

    if isinstance(threshold, (int, float)):
        th_high = np.abs(threshold)
        th_low = -th_high
    elif isinstance(threshold, list) and len(threshold) == 2:
        th_high = max(threshold)
        th_low = min(threshold)
    elif isinstance(threshold, np.ndarray) and threshold.size == 2:
        th_high = max(threshold)
        th_low = min(threshold)
    else:
        raise TypeError(f"Threshold of type '{type(threshold)}' not supported.")

    idx_data = np.where(np.logical_and(offsets > th_low, offsets < th_high))
    idx_norm = np.where(np.logical_or(offsets <= th_low, offsets >= th_high))

    if idx_norm[0].size == 0:
        return offsets, mz, None

    offsets = offsets[idx_data]
    data = mz[idx_data]
    norm = mz[idx_norm]

    return offsets, data, norm


def plot_z(mz: np.array,
           offsets: np.array = None,
           normalize: bool = False,
           norm_threshold: Union[int, float, list, np.ndarray] = 295,
           invert_ax: bool = True,
           plot_mtr_asym: bool = False,
           title: str = 'spectrum',
           x_label: str = 'offsets [ppm]',
           y_label: str = 'signal',
           **kwargs) \
        -> Figure:
    """
    initiating calculations and plotting functions
    :param mz: y-values
    :param offsets: x-values (e.g. offsets, inversion times)
    :param normalize: boolean to activate normalization
    :param norm_threshold: threshold for splitting into data and normalization data
    :param invert_ax: boolean to invert x-axis
    :param plot_mtr_asym: boolean to enable/disable plotting of MTRasym
    :param title: optional title for the plot
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    """
    if offsets is None:
        offsets = range(len(mz))

    if normalize:
        mz, offsets = normalize_data(mz, offsets, norm_threshold)

    fig, ax1 = plt.subplots()
    ax1.set_ylim([round(min(mz) - 0.05, 2), round(max(mz) + 0.05, 2)])
    ax1.set_ylabel(y_label, color='b')
    ax1.set_xlabel(x_label)
    plt.plot(offsets, mz, '.--', label='$Z$', color='b')
    if invert_ax:
        plt.gca().invert_xaxis()
    ax1.tick_params(axis='y', labelcolor='b')

    if plot_mtr_asym:
        mtr_asym = calc_mtr_asym(z=mz, offsets=offsets)

        ax2 = ax1.twinx()
        ax2.set_ylim([round(min(mtr_asym) - 0.01, 2), round(max(mtr_asym) + 0.01, 2)])
        ax2.set_ylabel('$MTR_{asym}$', color='y')
        ax2.plot(offsets, mtr_asym, label='$MTR_{asym}$', color='y')
        ax2.tick_params(axis='y', labelcolor='y')
        fig.tight_layout()

    plt.title(title)
    plt.show()

    return fig
