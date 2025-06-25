from collections import namedtuple
from pathlib import Path
import numpy as np

ROOT_PATH = Path(__file__).resolve().parent.parent
IMG_PATH = ROOT_PATH / "img"

GRID: np.ndarray = np.linspace(100., 0., num=101)
PFGRID: np.ndarray = np.array([100., 20., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.])

Atmosphere = namedtuple(
    'Atmosphere', ['aer_type', 'aer_profile', 'tau_aer', 'tau_ray', 'hmin']
)