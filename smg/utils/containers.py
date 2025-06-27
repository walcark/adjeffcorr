from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from smartg.smartg import Sensor


Atmosphere = namedtuple('Atmosphere', ['aer_type', 'aer_profile', 
                                       'tau_aer', 'tau_ray', 
                                       'hmin'])

Radiatives = namedtuple("Radiatives", ('rho_atm', 'tdir_up', 
                                       'tdir_down', 'tdif_up', 
                                       'tdif_down', 'sph_alb'))


@dataclass 
class SunSat: 
    sza_deg: float = 0.0
    saa_deg: float = 0.0
    vza_deg: float = 0.0
    vaa_deg: float = 0.0
    sat_height: float = 700.0

    @property
    def sun_le(self) -> Dict[str, float]:
        return self._le(self.sza_deg, self.saa_deg)
    
    @property
    def sat_le(self) -> Dict[str, float]:
        return self._le(self.vza_deg, self.vaa_deg)
    
    @property
    def sun_sensor(self) -> Sensor:
        return self._sensor(self.sat_height, self.sza_deg, self.saa_deg)

    @property
    def sat_sensor(self) -> Sensor:
        return self._sensor(self.sat_height, self.vza_deg, self.vaa_deg)
    
    @property
    def satellite_relative_position(self) -> Tuple[float, float]:
        """
        Returns the relative position (x, y) of a satellite from 
        the point P(x=0, y=0, z=0) it is looking at on the earth.
        """
        tan_vza: float = np.tan(np.radians(self.vza_deg))
        cos_vaa: float = np.cos(np.radians(180 - self.vaa_deg))
        sin_vaa: float = np.sin(np.radians(180 - self.vaa_deg))
        x: float  = (self.sat_height * tan_vza) * cos_vaa
        y: float = (self.sat_height * tan_vza) * sin_vaa
        return (np.round(x, 4), np.round(y, 4))

    def _le(self, za: float, aa: float) -> Dict[str, float]:
        return {"th_deg": za, "phi_deg": aa, "zip": True}

    def _sensor(self, height: float, za: float, aa: float) -> Sensor:
        return Sensor(POSZ=height, THDEG=180.0 - za, PHDEG=aa, LOC='ATMOS')                   

    
    def __str__(self) -> str:
        return "-".join([
            f"SZA-{self.sza_deg}", f"SAA-{self.saa_deg}",
            f"VZA-{self.vza_deg}", f"VAA-{self.vaa_deg}",
            f"H-{self.sat_height}"
        ])
