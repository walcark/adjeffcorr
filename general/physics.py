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
    surface_mot: float, 
    h: float | np.ndarray, 
    href: float
) -> float | np.ndarray:
    """
    Returns the MOT for the height h, given a surface MOT and a 
    reference height for the exponential model of the SOS_ABS_V5.1
    model.
    """
    return surface_mot * np.exp(-h / href)


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
    wavelength = np.asarray([wl])
    h = np.asarray(h)
    
    # Préparer la sortie avec les dimensions appropriées
    out = np.zeros((len(wavelength), len(h)))
    
    # Calculer pour chaque longueur d'onde et chaque altitude
    for i, wl in enumerate(wavelength):
        surf_mot = surface_mot_sos(wavelength)
        out[i, :] = height_mot_sos(surf_mot, h, href)

    # Calcul de la différence colonne à colonne
    out_diff = np.zeros((len(wavelength), len(h)))
    for j in range(len(h)-1):
        out_diff[:, j+1] = out[:, j+1] - out[:, j]

    return out_diff
