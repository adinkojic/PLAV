import json
import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84
import brgr_aero_forces_linearized as aero
from atmosphere import Atmosphere
import aircraftconfig

wind_alt_profile = np.array([0, 10000], dtype='d')
wind_speed_profile = np.array([0.0, 0.0], dtype='d')
wind_direction_profile = np.array([0, 0], dtype='d')
#init atmosphere config
atmosphere = Atmosphere(wind_alt_profile,wind_speed_profile,wind_direction_profile)

altitude = 1

atmosphere.update_conditions(altitude, 0)

velocity = np.array([10, 100, 0.1])
air_velocity = velocity + atmosphere.get_wind_ned()
print(velocity)
print(air_velocity)
print(aircraftconfig.velocity_to_alpha_beta(air_velocity))


#load aircraft config
with open('aircraftConfigs/sphere.json', 'r') as file:
    modelparam = json.load(file)
file.close()

aircraft = aircraftconfig.init_aircraft(modelparam)
aircraft.update_conditions(altitude,  air_velocity, np.array([0.0,0.0,0.0]), 1.225, 288.15)


body_forces_body, moments = aircraft.get_forces()
print(body_forces_body)
