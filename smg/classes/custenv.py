from smartg.smartg import Albedo_cst, Albedo_map, Environment
import numpy as np

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
    