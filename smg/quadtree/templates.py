from smartg.smartg import LambSurface, Environment, Sensor, Smartg
from smartg.atmosphere import AtmAFGL
from smg.classes.sunsat import SunSat
from typing import List, Tuple
from luts.luts import MLUT
import numpy as np


def rho_toa(
    points: List[Tuple],
    wavelength: float,
    atmosphere: AtmAFGL,
    surface: LambSurface,
    environment: Environment,
    sunsat: SunSat = SunSat(), 
    nb_photons: int = 1E5, 
    nb_angles: int = 1E4, 
    **kwargs
) -> np.ndarray:
    """
    Calculates the Top Of Atmosphere reflectance (TOA) 
    for the given atmosphere and environment on a 2D grid.
    """
    print(f"Calling rho_toa with {len(points)} points ...")
    # Grid of satellite positions
    x_sat, y_sat = sunsat.satellite_relative_position
    sat_locs = [(x + x_sat, y + y_sat) for (x, y) in points]

    # Total number of photons to launch
    nb_photons_tot: int = len(sat_locs) * nb_photons

    # Top-of-atmosphere sensors to launch photons
    sensors_toa: List[Sensor] = [
        Sensor(
            POSX=pos_x, 
            POSY=pos_y, 
            POSZ=sunsat.sat_height,
            THDEG=180-sunsat.vza_deg,
            PHDEG=sunsat.vaa_deg,
            LOC='ATMOS'
        ) for (pos_x, pos_y) in sat_locs
    ]
    
    # Computation with Smart-G and extraction of e_coupling
    sg = Smartg(back=True)
    mlut: MLUT = sg.run(
        wl=wavelength, 
        atm=atmosphere,
        surf=surface, 
        env=environment,
        sensor=sensors_toa,
        le=sunsat.sun_le,
        NBPHOTONS=nb_photons_tot,
        NF=nb_angles,
    )
    print("Finished rho_toa computation.")
    return list(mlut["I_up (TOA)"][:, 0])


def rho_coupling(
    points: List[Tuple],
    wavelength: float,
    atmosphere: AtmAFGL,
    surface: LambSurface,
    environment: Environment,
    sunsat: SunSat = SunSat(),
    nb_photons: int = 1E5, 
    nb_angles: int = 1E4, 
    **kwargs
) -> np.ndarray:
    """
    Direct calculation of 'E_coupling' for a given 
    atmosphere and environment on a 2D grid.
    """
    print(f"Calling rho_coupling with {len(points)} points ...")

    lamb_emit_boa: List[Sensor] = [
        Sensor(
            POSX=pos_x, 
            POSY=pos_y, 
            POSZ=0.0,
            FOV=90.,
            TYPE=1,
            LOC='ATMOS'
        ) for (pos_x, pos_y) in points
    ]

    mlut: MLUT = Smartg(back=True).run(
        wl=wavelength,
        atm=atmosphere,
        surf=surface,
        env=environment,
        sensor=lamb_emit_boa,
        le=sunsat.sun_le,
        NBPHOTONS=len(points) * nb_photons,
        NF=nb_angles,
    )
    e_coupling: np.ndarray = mlut["I_up (TOA)"][:, 0]

    print("Finished rho_coupling computation.")
    return e_coupling

