from typing import List, Tuple
import numpy as np


def gaussian(
    coords: List[Tuple], 
    sigma: float
) -> List[float]:
    """
    Simple gaussian function for test purpose.

    Params:
        * coords (List[Tuple[float, float]]):
            list of points coordinates
        * sigma (float): 
            standard deviation of the gaussian

    Returns:
        (List[float]) the values of the gaussian for coords
    """
    arr = np.sqrt(np.sum(np.square(coords), axis=1))
    return np.exp(-(arr/sigma)**2).tolist()


def smooth_disk(
    coords: List[Tuple[float, float]],
    rmax: float,
    smooth_factor: float = 10.0
) -> List[float]:
    """
    Simple disk function with a smooth frontier for test purpose.

    Params:
        * coords (List[Tuple[float, float]]):
            list of points coordinates
        * rmax (float): 
            radius of the disk
        * smooth_factor (float):
            smoothing factor for the disk frontier transition

    Returns:
        (List[float]) the values of the smoothed disk for coords
    """
    arr = np.asarray(coords, dtype=float)      
    r = np.linalg.norm(arr, axis=1) 
    val = 1.0 / (1.0 + np.exp(smooth_factor * (r - rmax)))
    return val.tolist()


if __name__ == "__main__":

    from smg.general.plot1d import plot_scientific

    sigma = 5.0
    rmax = 15.0
    smooth_factor = 1

    coord_vals = np.linspace(-50, 50, 1000)
    coords = [(x, y) for (x, y) in zip(coord_vals, coord_vals)]
    radius = [np.sqrt(x**2 + y**2) for (x, y) in zip(coord_vals, coord_vals)]

    gauss_vals = gaussian(coords, sigma)
    disk_vals = smooth_disk(coords, rmax, smooth_factor)

    plot_scientific(
        x=radius, ys=[gauss_vals, disk_vals],
        labels=[r"Gaussian: $\sigma = 5$", r"Disk: $r_{\max} = 15$"],
        xlabel="Distance [km]",
        ylabel="Normalized value",
        title="Comparison of Radial Kernels"
    )
