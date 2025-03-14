import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84
import brgr_aero_forces_linearized as aero
from atmosphere import Atmosphere

wind_alt_profile = np.array([0, 10000], dtype='d')
wind_speed_profile = np.array([6.096, 6.096], dtype='d')
wind_direction_profile = np.array([270, 270], dtype='d')
#init atmosphere config
atmosphere = Atmosphere(wind_alt_profile,wind_speed_profile,wind_direction_profile)

altitude = 1

atmosphere.update_conditions(altitude, 0)

velocity = np.array([0, 6 ,0])
print(velocity+ atmosphere.get_wind_ned())