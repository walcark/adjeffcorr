from matplotlib.ticker import AutoMinorLocator
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

def plot_scientific(
    x: np.ndarray,
    ys: List[np.ndarray],
    labels: Optional[List[str]] = None,
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    yscale: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 5),
    colors: Optional[List[str]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Plot one or more 1D curves using a clean scientific style.

    Args:
        x (np.ndarray): X-axis values.
        ys (List[np.ndarray]): List of y-axis data series.
        labels (List[str], optional): Labels for each data series.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str, optional): Plot title.
        xlim (tuple, optional): X-axis limits.
        ylim (tuple, optional): Y-axis limits.
        figsize (tuple): Figure size in inches.
        colors (list, optional): Custom colors for each curve.
        savepath (str, optional): Path to save the figure.
        show (bool): Whether to display the plot immediately.
    """
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=figsize)

    for i, y in enumerate(ys):
        label = labels[i] if labels and i < len(labels) else None
        color = colors[i] if colors and i < len(colors) else None
        ax.plot(x, y, label=label, lw=2, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(np.min(x), np.max(x))
    if (yscale == "log"):
        ax.set_yscale("log")
    else:
        if ylim:
            ax.set_ylim(*ylim)
        else:
            mini = np.min([np.min(vals) for vals in ys])
            maxi = 1.1 * np.max([np.max(vals) for vals in ys])
            ax.set_ylim(mini, maxi)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", length=6)
    ax.tick_params(axis="both", which="minor", length=3)

    if labels:
        ax.legend(loc="best")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath)

    if show:
        plt.show()
