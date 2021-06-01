"""
eval.py
    Tool independent functions for plotting and calculations
"""
import numpy as np
import matplotlib.pyplot as plt
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


def plot_z(mz: np.array,
           offsets: np.array = None,
           invert_ax: bool = True,
           plot_mtr_asym: bool = False,
           title: str = None,
           x_label: str = 'offset',
           y_label:str = 'normalized signal') \
        -> Figure:
    """
    initiating calculations and plotting functions
    :param mz: magnetization vector
    :param offsets: offsets to plot the magnetization on
    :param invert_ax: boolean to invert x-axis
    :param plot_mtr_asym: boolean to enable/disable plotting of MTRasym
    :param title: optional title for the plot
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    """
    if offsets is None:
        offsets = range(len(mz))

    if title is None:
        title = 'Z-Spec'

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
