import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84
import brgr_aero_forces_linearized as aero

from aircraftconfig import AircraftConfig

@jit
def get_gravity(phi, h):
    """gets gravity accel from lat and altitude
    phi: latitude
    h: altitude"""
    graivty = 9.780327*(1 +5.3024e-3*np.sin(phi)**2 - 5.8e-6*np.sin(2*phi)**2) \
            - (3.0877e-6 - 4.4e-9*np.sin(phi)**2)*h + 7.2e-14*h**2
    return graivty

phi = 0.698132

print(get_gravity(phi, 0))
print(get_gravity(phi, 9000))
