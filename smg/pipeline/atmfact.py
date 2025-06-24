from smartg.atmosphere import AtmAFGL, AerOPAC
from dataclasses import dataclass
from typing import List, Any
import numpy as np

from general.physics import molecular_profile
from general.constants import GRID, PFGRID

#TODO: make use of multi_profiles in smartg.py

@dataclass
class AtmFact:
    aot: float
    wl: float
    w_ref: float = 550.0
    aer_type: str = "continental_average"
    atmo_type: str = "afglt"
    grid: np.ndarray = GRID
    pfgrid: np.ndarray = PFGRID

    def __post_init__(self) -> None:
        self._params = {
            "wl": self.wl,
            "aot": self.aot,
            "aer_type": self.aer_type,
            "atmo_type": self.atmo_type,
        }
        self.atm = self._build_atmosphere()

    def create_standard_atm(self) -> AtmAFGL:
        """Returns a standard atmosphere (with complete absorption)."""
        return self.atm

    def create_atm_no_abs(self) -> AtmAFGL:
        """Returns an atmosphere without gaseous absorption."""
        atm_modified = self._build_atmosphere(rm_gaz_abs=True)
        print("Created atmosphere without gaseous absorption.")
        return atm_modified

    def create_atm_no_abs_sos_ray(self) -> AtmAFGL:
        cust_pro_ray = molecular_profile(wl=self.wl, h=GRID, href=8.0)
        atm_modified = self._build_atmosphere(cust_pro_ray, rm_gaz_abs=True)
        print("Created atmosphere without gaseous absorption.")
        return atm_modified

    def create_atm_no_abs_and_ray(self) -> AtmAFGL:
        """Returns an atmosphere without gaseous and Rayleigh absorption."""
        atm_modified = self._build_atmosphere(rm_gaz_abs=True, rm_ray_abs=True)
        print("Created atmosphere without gaseous / Rayleigh absorption.")
        return atm_modified

    def _build_aer_list(self) -> List[AerOPAC]:
        """Creates the list of AerOPAC components for each aot value."""
        return [AerOPAC(self.aer_type, self.aot, self.w_ref)]

    def _build_atmosphere(
            self,
            cust_ray: np.ndarray = None,
            **profile_modifications: Any, 
        ) -> AtmAFGL:
        """
        Constructs an AtmAFGL instance, optionally applying modifications to the profiles.
        
        Expected keyword arguments:
         - rm_gaz_abs: if True, removes gaseous absorption.
         - rm_ray_abs: if True, removes Rayleigh absorption.
        """
        if profile_modifications:
            abs, ray, (aer, ssa_aer), phases = self.atm.calc_split(self.wl)
            if (cust_ray is not None): 
                ray = cust_ray
            if profile_modifications.get('rm_ray_abs', False):
                abs = np.zeros_like(abs)
            if profile_modifications.get('rm_gaz_abs', False):
                ssa_aer = np.ones_like(ssa_aer)
            return AtmAFGL(
                self.atmo_type,
                pfwav=self.wl,
                prof_abs=abs,
                prof_ray=ray,
                prof_aer=(aer, ssa_aer),
                prof_phases=phases,
                grid=self.grid,
                pfgrid=self.pfgrid
            )
        else:
            aer_list = self._build_aer_list()
            return AtmAFGL(
                self.atmo_type,
                pfwav=self.wl,
                comp=aer_list,
                grid=self.grid,
                pfgrid=self.pfgrid
            )
