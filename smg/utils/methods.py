from typing import Union, List, Optional
from numpy.typing import ArrayLike
import numpy as np

from smartg.smartg import LambSurface, Environment, Sensor, Smartg, Albedo_cst, Albedo_map
from smartg.atmosphere import AtmAFGL, MLUT
from typing import List, Tuple
from luts.luts import MLUT

from smg.utils.containers import SunSat, Radiatives, Atmosphere


def image_to_atmospheres(
    tau_aer: ArrayLike, 
    tau_ray: ArrayLike, 
    mnt: ArrayLike, 
    aer_type: str, 
    aer_profile: str
) -> List[Atmosphere]:
    """
    """
    shp = tau_aer.shape
    if (tau_ray.shape != shp) or (mnt.shape != shp):
        raise ValueError("Input array shape should be the same.")
    
    atms = []
    for a, r, h in zip(tau_aer.flat, tau_ray.flat, mnt.flat):
        atms.append(Atmosphere(
            aer_type=aer_type, 
            aer_profile=aer_profile,
            tau_aer=a, 
            tau_ray=r, 
            hmin=h)
        )
    return atms



def custom_environment(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    rhos: np.ndarray,
    n_alb: int = 10 
) -> None:
    """
    Creates a custom Environment for a complex rho_s scene.

    Args:
        surface_reflectance (Field): _description_
        sunsat (SunSat): _description_
        atm (AtmAFGL): _description_
        wl (float): _description_
    """
    # Reduction of the albedo map
    mini = np.min(rhos)
    maxi = np.max(rhos)
    albs = np.linspace(mini, maxi, n_alb)
    nx, ny = rhos.shape
    rhos_idx = np.zeros((nx+1, ny+1))
    for i in range(nx):
        for j in range(ny):
            rhos_idx[i+1, j+1] = np.sum(rhos[i, j] > albs)
    albs = [Albedo_cst(np.round(alb, 2)) for alb in albs]   
    
    # Environment object creation
    x: np.ndarray = np.append(x_coords, 1E8)
    y: np.ndarray = np.append(y_coords, 1E8) 
    ALB = Albedo_map(rhos_idx, x, y, albs)
    env = Environment(ENV=5, ALB=ALB)
    return env
    


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


class RadiativesFactory:
    """
    Class to allow calculation of radiative properties with Smart-G, either 
    for single atmosphere or multiple atmospheres (for instance a on a full
    Sentine-2 image).
    """
    def __init__(self, sunsat: SunSat = SunSat(), NF: int = 1E4):
        self.sunsat = sunsat
        self.NF = NF
        self.smartg = Smartg()

    def rho_atm(self, atm: Union[AtmAFGL, MLUT]) -> Union[float, np.ndarray]:
        """
        Launch a Smart-G simulation to calculate the atmospheric reflectance.
        Geometry:   Satellite ----> Atmosphere only (black ground) ---> Sun
        """
        mlut = self._simulate(
            sensor=self.sunsat.sat_sensor, 
            le=self.sunsat.sun_le,
            atm=atm
        )
        mlut.describe()
        print(mlut["I_up (TOA)"][:, 0].shape)
        return mlut["I_up (TOA)"][:, 0]

    def optical_depth(self, atm: Union[AtmAFGL, MLUT]) -> Union[float, np.ndarray]:
        """
        Return the optical depth of the given atmosphere.
        """
        mlut = self._simulate(NBPHOTONS=1e2, atm=atm)
        return float(mlut["OD_atm"][-1])

    def tdir_up(self, atm: Union[AtmAFGL, MLUT]) -> Union[float, np.ndarray]:
        """
        Calculates the direction upward transmittance of the 
        given atmosphere.
        """
        od = self.optical_depth()
        vza_rad = np.radians(self.sunsat.vza_deg)
        return np.exp(-od / np.cos(vza_rad))

    def tdir_down(self, atm: Union[AtmAFGL, MLUT]) -> Union[float, np.ndarray]:
        """
        Calculates the direction upward transmittance of the 
        given atmosphere.
        """
        od = self.optical_depth()
        sza_rad = np.radians(self.sunsat.sza_deg)
        return np.exp(-od / np.cos(sza_rad))

    def tdif_up(self, atm: Union[AtmAFGL, MLUT]) -> Union[float, np.ndarray]:
        """
        Launch a Smart-G simulation to calculate the upward diffuse 
        transmittance.
        Geometry:   Ground sensor ---> Satellite local estimate.
        """
        mlut = self._simulate(
            sensor=Sensor(POSZ=0.0, LOC='ATMOS', TYPE=1, FOV=90), 
            le=self.sunsat.sat_le,
            atm=atm
        )
        return float(mlut["I_up (TOA)"][0])

    def tdif_down(self, atm: Union[AtmAFGL, MLUT]) -> Union[float, np.ndarray]:
        """
        Launch a Smart-G simulation to calculate the downward diffuse 
        transmittance.
        Geometry:   Sun sensor ---> Ground (flux)
        """
        mlut = self._simulate(
            sensor=self.sunsat.sun_sensor,
            OUTPUT_LAYERS=3,
            flux='planar',
            atm=atm
        )
        return float(mlut['flux_down (0+)'].to_xarray())

    def spherical_albedo(self, atm: Union[AtmAFGL, MLUT]) -> Union[float, np.ndarray]:
        """
        Launch a Smart-G simulation to calculate the spherical albedo 
        of the atmosphere.
        Geometry:   Ground sensor ---> Ground (flux)
        """
        mlut = self._simulate(
            sensor=Sensor(POSZ=0.0, LOC='ATMOS', TYPE=1, FOV=90),
            OUTPUT_LAYERS=3,
            flux='planar',
            atm=atm
        )
        return float(mlut['flux_down (0+)'].to_xarray())

    def run_all(self, atm: Union[AtmAFGL, MLUT]) -> Radiatives:
        """Returns all the atmospheric properties."""
        return Radiatives(
            rho_atm=self.rho_atm(atm), 
            tdir_up=self.tdir_up(atm), 
            tdif_up=self.tdif_up(atm), 
            tdir_down=self.tdir_down(atm), 
            tdif_down=self.tdif_down(atm), 
            sph_alb=self.spherical_albedo(atm)
        )
    
    def _simulate(
        self, 
        atm: Union[AtmAFGL, MLUT], 
        wl: Optional[Union[float, List[float]]] = None, 
        NPH: int = 1E9,
        **kwargs
    ) -> MLUT:
        """
        Run the Smart-G simulation for the atmosphere provided.
        """
        if (wl):
            if (isinstance(atm, MLUT)):
                assert (atm.axes["wavelength"] == wl)
            else:
                atm = atm.calc(wl)
        else:
            if (isinstance(atm, AtmAFGL)):
                raise ValueError("wl should be defined to compute atm LUT.")
            wl = atm.axes["wavelength"]

        default_params = {'wl': wl, 'atm': atm, 'NBPHOTONS': NPH, 'NF': self.NF}
        default_params.update(kwargs)
        mlut = self.smartg.run(**default_params)
        return mlut