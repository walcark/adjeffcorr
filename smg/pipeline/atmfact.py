from typing import List, Tuple
import numpy as np
from numpy.typing import ArrayLike
from smartg.atmosphere import AtmAFGL, AerOPAC
from smartg.smartg import multi_profiles
from general.physics import molecular_profile
from luts.luts import LUT
from smartg.smartg import Smartg
from smartg.tools.smartg_view import input_view
import matplotlib.pyplot as plt

from general.constants import Atmosphere, GRID, PFGRID


class AtmAFGLFactory:
    """
    Factory for building and grouping atmospheric LUTs (AtmAFGL) across
    varying aerosol optical thickness and surface elevation (hmin).
    """
    def __init__(
        self,
        wl: ArrayLike,
        grid_size: int = 100,
        pfgrid_size: int = 10,
    ) -> None:
        """
        Initialize with wavelengths and grid parameters.
        :param wl: wavelengths (scalar or array-like)
        :param grid_size: number of levels for altitude grid
        :param pfgrid_size: number of levels for fine grid
        :param max_bins: max groups by elevation
        """
        self.wl = np.asarray(wl, dtype=float)
        self.atms: List[Atmosphere] = []
        self.grid_size = grid_size
        self.pfgrid_size = pfgrid_size

    def compute(self, calls: List[Atmosphere], **opt) -> LUT:
        luts = []
        for call in calls:
            afgl: AtmAFGL = self._atmafgl(call, **opt)
            luts.append(afgl.calc(self.wl))
        return multi_profiles(luts, kind="atm")

    def _atmafgl(self, atm: Atmosphere, **opt) -> AtmAFGL:
        # optional arguments
        maja_mode = opt.get('maja', False)
        mol_abs = opt.get("molecular_absorption", True)
        # preparing the atmosphere description parameters
        aer_type, aer_profile, tau_aer, tau_ray, hmin = tuple(atm)
        grid = GRID + hmin
        pfgrid = PFGRID + hmin
        # creating the AerOPAC instance
        aer_kwargs = {}
        if (maja_mode):
            aer_kwargs = dict(
                H_mix_min = hmin, H_mix_max = hmin + 2000.0,
                H_stra_min = 0.0, H_stra_max = 0.0,
                H_free_min = 0.0, H_free_max = 0.0
            )
        aer = AerOPAC(aer_type, tau_aer, 550.0, **aer_kwargs)
        # creating the afgl instance
        atm_kwargs = dict(
            atm_filename=aer_profile,
            comp=[aer],
            grid=grid,
            pfgrid=pfgrid,
            tauR=tau_ray
        )
        atm_obj = AtmAFGL(**atm_kwargs)
        if maja_mode or not(mol_abs):
            abs_prof, ray_prof, (a_p, ssa), phases = atm_obj.calc_split(self.wl)
            if (maja_mode):
                ray_prof = molecular_profile(self.wl, grid, href=8.0)
            if not(mol_abs):
                abs_prof = np.zeros_like(abs_prof)
            atm_obj = AtmAFGL(
                prof_abs=abs_prof,
                prof_ray=ray_prof,
                prof_aer=(a_p, ssa),
                prof_phases=phases,
                **atm_kwargs
            )  
        return atm_obj        

if __name__ == "__main__":
    from smg.pipeline.preprocessor import Preprocessor

    # data definition
    wl = [443.0, 490.0, 560.0]
    atm_type = "continental_average"
    aer_profile = "afglms"
    data = [
        Atmosphere(atm_type, aer_profile, 0.0,  0.0, 0.05),
        Atmosphere(atm_type, aer_profile, 0.0,  0.00, 0.1),
        Atmosphere(atm_type, aer_profile, 0.0,  0.0, 0.5),
        Atmosphere(atm_type, aer_profile, 0.5,  0.0, 0.9),
        Atmosphere(atm_type, aer_profile, 0.5,  0.0, 1.2),
        Atmosphere(atm_type, aer_profile, 0.0,  0.5, 1.7),
        Atmosphere(atm_type, aer_profile, 0.0,  0.5, 2.1),
        Atmosphere(atm_type, aer_profile, 0.5,  0.5, 4.1),
    ]

    # grouping data
    preproc: Preprocessor = Preprocessor(Atmosphere, "hmin", 1.0)
    groups = preproc.get_groups(data)
    for group in groups:
        print(f"Group ({preproc.group_key}={preproc._get_value(group[0], preproc.group_key)}):")
        for item in group:
            print("  ", item)

    # building atmospheres
    factory: AtmAFGLFactory = AtmAFGLFactory(wl)
    afgl_list = []
    for group in groups:
        afgl = factory.compute(group)
        afgl_list.append(afgl)

        m = Smartg(autoinit=True).run(wl=afgl.axes["wavelength"], THVDEG=60., atm=afgl, NBPHOTONS=1e9)
        print(m)
    #input_view(m, kind='atm', zmax=50)
    #plt.show()

