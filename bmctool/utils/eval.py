"""
eval.py
    Tool independent functions for plotting and calculations
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def calc_mtr_asym(z: np.ndarray) -> np.ndarray:
    """
    calculating MTRasym from the magnetization vector
    :param z: magnetization
    :return: MTRasym
    """
    return np.flip(z) - z


def plot_z(mz: np.array,
           offsets: np.array = None,
           invert_ax: bool = True,
           plot_mtr_asym: bool = False,
           title: str = None) \
        -> Figure:
    """
    initiating calculations and plotting functions
    :param mz: magnetization vector
    :param offsets: offsets to plot the magnetization on
    :param invert_ax: boolean to invert x-axis
    :param plot_mtr_asym: boolean to enable/disable plotting of MTRasym
    :param title: optional title for the plot
    """
    if offsets is None:
        offsets = range(len(mz))

    if title is None:
        title = 'Z-Spec'

    fig, ax1 = plt.subplots()
    ax1.set_ylim([0, 1.05])
    ax1.set_ylabel('Z', color='b')
    ax1.set_xlabel('Offsets')
    plt.plot(offsets, mz, '.--', label='$Z$', color='b')
    if invert_ax:
        plt.gca().invert_xaxis()
    ax1.tick_params(axis='y', labelcolor='b')

    if plot_mtr_asym:
        mtr_asym = calc_mtr_asym(mz)

        ax2 = ax1.twinx()
        ax2.set_ylim([0, round(np.max(mtr_asym) + 0.01, 2)])
        ax2.set_ylabel('$MTR_{asym}$', color='y')
        ax2.plot(offsets, mtr_asym, label='$MTR_{asym}$', color='y')
        ax2.tick_params(axis='y', labelcolor='y')
        fig.tight_layout()

    plt.title(title)
    plt.show()
    return fig
