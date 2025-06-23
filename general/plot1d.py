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
    ax: Optional[plt.Axes] = None,
):
    """
    Plot one or more 1D curves using a clean scientific style.

    If `ax` is provided, plot on that axes. Otherwise, create a new figure.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 12,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "figure.dpi": 150,
    })

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    for i, y in enumerate(ys):
        label = labels[i] if labels and i < len(labels) else None
        color = colors[i] if colors and i < len(colors) else None
        ax.plot(x, y, label=label, lw=2, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(np.min(x), np.max(x))
    if yscale == "log":
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

    if savepath and created_fig:
        fig.savefig(savepath)
    if show and created_fig:
        plt.show()
