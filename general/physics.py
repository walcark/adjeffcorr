import numpy as np

def surface_mot_sos(
    wl: float | np.ndarray
) -> float | np.ndarray:
    """
    Returns the surface MOT (Molecular Optical Thickness) used for 
    a given wavelength by the SOS_ABS_V5.1 model.
    """
    wlm = wl / 1000.0
    return 1E-4 * (84.35 / wlm ** 4 - 1.225 / wlm ** 5 + 1.4 / wlm ** 6)


def height_mot_sos(
    surface_mot: float | np.ndarray, 
    h: float | np.ndarray, 
    href: float
) -> float | np.ndarray:
    """
    Returns the MOT for the height h, given a surface MOT and a 
    reference height for the exponential model of the SOS_ABS_V5.1
    model.
    """
    print(surface_mot.shape)
    print(h.shape)
    return surface_mot[:, None] * np.exp(-h[None, :] / href)


def molecular_profile(
    wl: float | np.ndarray, 
    h: float | np.ndarray, 
    href: float
) -> np.ndarray:
    """
    Return the molecular profile of the SOS_ABS_5.1 model to provide 
    as input to Smart-G for a given wavelength, height array and 
    reference height.
    """
    if isinstance(wl, float):
        wl = [wl]
    if isinstance(h, float):
        h = [h]
    wl = np.asarray(wl)
    h = np.asarray(h)

    surface_mot = surface_mot_sos(wl)
    full_mot = height_mot_sos(surface_mot, h, href)
    full_kext = np.zeros_like(full_mot)

    for j in range(len(h)-1):
        full_kext[:, j+1] = full_mot[:, j+1] - full_mot[:, j]

    return full_kext
