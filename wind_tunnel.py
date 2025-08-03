"""The "Wind Tunnel", where you can verify flight dynamics models.
Input speeds, conditions, angles-of-attack, and other data to get forces and moments"""

import math
import json
import numpy as np

#from plav import get_gravity

import ussa1976
import quaternion_math as quat

import brgrModel
import f16_model
import genericAircraftConfig
from genericAircraftConfig import velocity_to_alpha_beta, alpha_beta_to_velocity, get_wind_to_body_axis



RAD_TO_DEG = 180/math.pi
M_TO_FT = 39.37/12

alpha_deg = 11.0
beta_deg = 0.0
airspeed = 15.0
altitude = 0.0
body_rate = np.array([0.,0.,0.,],'d')


pdt = ussa1976.get_pressure_density_temp(altitude)


with open('aircraftConfigs/brgrDroneDrop.json', 'r') as file:
    modelparam = json.load(file)
file.close()
model = brgrModel.init_aircraft(modelparam)
S = modelparam['Sref']
cbar = modelparam['cref']
b = modelparam['bref']

alpha = alpha_deg / RAD_TO_DEG
beta  = beta_deg / RAD_TO_DEG
density = pdt[1]
speed_of_sound = ussa1976.get_speed_of_sound(pdt[2])

qbar = 0.5 * density * airspeed**2

wind_velocity = alpha_beta_to_velocity(airspeed, alpha, beta)
model.update_conditions(altitude, wind_velocity, body_rate, density, pdt[2], speed_of_sound)
model.update_control(np.array([0.0, 0.0, -1.1, 0.0],'d'))

print(f"Wind Velocity: {wind_velocity}")

forces, moments = model.get_forces()

wind_to_body_axis = get_wind_to_body_axis(alpha, beta)
forces = quat.rotateFrameQ(wind_to_body_axis, forces)

print("Forces:", forces)

coeff_forces = forces / qbar / S

print("Sref:", S)
print("cref:", cbar)
print("bref:", b)

coeff = model.get_coeff()

print(f"\n  Lift Coefficient: {coeff[0]:.4f}"
      f"\n  Drag Coefficient: {coeff[1]:.4f}"
      f"\n  Side Force Coefficient: {coeff[3]:.4f}"
      f"\n  Roll Moment Coefficient: {coeff[4]:.4f}"
      f"\n  Pitch Moment Coefficient: {coeff[2]:.4f}"
      f"\n  Yaw Moment Coefficient: {coeff[5]:.4f}")

print(f"Wind Tunnel Results at {airspeed} m/s, alpha={alpha_deg} deg, beta={beta_deg} deg, altitude={altitude} ft"
      f"\n  Lift Coefficient: {-coeff_forces[2]:.4f}"
      f"\n  Drag Coefficient: {-coeff_forces[0]:.4f}"
      f"\n  Side Force Coefficient: {coeff_forces[1]:.4f}"
      f"\n  Roll Moment Coefficient: {moments[0] / qbar / b:.4f}"
      f"\n  Pitch Moment Coefficient: {moments[1] / qbar / cbar:.4f}"
      f"\n  Yaw Moment Coefficient: {moments[2] / qbar / b:.4f}")

print(f"L/D {forces[2] / forces[0]:.4f}")