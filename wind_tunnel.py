"""The "Wind Tunnel", where you can verify flight dynamics models.
Input speeds, conditions, angles-of-attack, and other data to get forces and moments"""

import math
import numpy as np

#from plav import get_gravity

import ussa1976

import brgrModel
import f16_model
import genericAircraftConfig
from genericAircraftConfig import velocity_to_alpha_beta, alpha_beta_to_velocity



RAD_TO_DEG = 180/math.pi
M_TO_FT = 39.37/12

alpha_deg = 5.0
beta_deg = -2.340
airspeed = 300.0 / M_TO_FT
altitude = 0.0
body_rate = np.array([0.,0.,0.,],'d')

S = 300 / M_TO_FT**2
cbar = 11.32 / M_TO_FT
b = 30 / M_TO_FT

pdt = ussa1976.get_pressure_density_temp(altitude)

model = f16_model.F16_aircraft(np.array([0.0,0.0,0.0,0.0],'d'))

alpha = alpha_deg / RAD_TO_DEG
beta  = beta_deg / RAD_TO_DEG
density = pdt[1]
speed_of_sound = ussa1976.get_speed_of_sound(pdt[2])

qbar = 0.5 * density * airspeed**2

wind_velocity = alpha_beta_to_velocity(airspeed, alpha, beta)
model.update_conditions(altitude, wind_velocity, body_rate, density, pdt[2], speed_of_sound)

forces, moments = model.get_forces()

coeff_forces = forces / qbar / S

print(model.get_coeff())
