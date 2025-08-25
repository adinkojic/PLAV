"""The "Wind Tunnel", where you can verify flight dynamics models.
Input speeds, conditions, angles-of-attack, and other data to get forces and moments"""

import math
import json
import numpy as np

#from plav import get_gravity

from plav.atmosphere_models.ussa1976 import get_pressure_density_temp, get_speed_of_sound
from plav.quaternion_math import rotateFrameQ
from plav.vehicle_models.generic_aircraft_config import AircraftConfig
from plav.vehicle_models.generic_aircraft_config import velocity_to_alpha_beta, alpha_beta_to_velocity, get_wind_to_body_axis

from plav.plav import load_scenario, load_aircraft_config

import plav.conversions as conv

class WindTunnel(object):
    """Wind Tunnel, for testing flight dynamics models"""
    def __init__(self, scenario_file):

        modelparam = load_scenario("scenarios/" + scenario_file)
        self.model, _ = load_aircraft_config(modelparam)
        self.S = modelparam['Sref']
        self.b = modelparam['bref']
        self.cbar = modelparam['cref']
        self.alpha = 0.0
        self.beta = 0.0
        self.airspeed = 1.0
        self.body_rate = np.array([0.,0.,0.,],'d')
        self.altitude = 0.0
        self.pdt = get_pressure_density_temp(self.altitude)
        self.density = self.pdt[1]
        self.qbar = 0.0
        self.update_qbar()
        self.speed_of_sound = 343.0 #ion wanna deal with this

    def change_alpha(self, alpha_deg):
        """Model alpha [deg]"""
        self.alpha = alpha_deg / conv.RAD_TO_DEG

    def change_beta(self, beta_deg):
        """Model beta [deg]"""
        self.beta = beta_deg / conv.RAD_TO_DEG

    def change_airspeed(self, airspeed):
        """Model airspeed [m/s]"""
        self.airspeed = airspeed
        self.update_qbar()

    def change_body_rate(self, body_rate):
        """Model body rate [rad/s]"""
        self.body_rate = body_rate

    def change_altitude(self, altitude):
        """Model altitude [m]"""
        self.altitude = altitude
        self.pdt = get_pressure_density_temp(altitude)
        self.density = self.pdt[1]
        self.update_qbar()

    def update_qbar(self):
        """Called internallly"""
        self.qbar = 0.5 * self.density * self.airspeed**2

    def solve_forces(self, quiet = False):
        """Solve and print forces"""
        wind_velocity = alpha_beta_to_velocity(self.airspeed, self.alpha, self.beta)
        self.model.update_conditions(self.altitude, wind_velocity, self.body_rate, \
        self.density, self.pdt[2], self.speed_of_sound)
        self.model.update_control(np.array([0.0, 0.0, 0.0, 0.0],'d'))
        forces, moments = self.model.get_forces()

        
        wind_to_body_axis = get_wind_to_body_axis(self.alpha, self.beta)
        forces = rotateFrameQ(wind_to_body_axis, forces)
        coeff_forces = forces / self.qbar / self.S
        
        coeff = self.model.get_coeff()
        
        if not quiet:
            print(f"Wind Velocity: {wind_velocity}")
            print("Forces:", forces)

            print("Sref:", self.S)
            print("cref:", self.cbar)
            print("bref:", self.b)

            print(f"Wind Tunnel Results at {self.airspeed} m/s, \
                alpha={self.alpha * conv.RAD_TO_DEG} deg, \
                beta={self.beta * conv.RAD_TO_DEG} deg, altitude={self.altitude} ft"
            f"\n  Lift Coefficient: {-coeff_forces[2]:.4f}"
            f"\n  Drag Coefficient: {-coeff_forces[0]:.4f}"
            f"\n  Side Force Coefficient: {coeff_forces[1]:.4f}"
            f"\n  Roll Moment Coefficient: {moments[0] / self.qbar / self.b:.4f}"
            f"\n  Pitch Moment Coefficient: {moments[1] / self.qbar / self.cbar:.4f}"
            f"\n  Yaw Moment Coefficient: {moments[2] / self.qbar / self.b:.4f}")

            print(f"Body Forces:")
            print(f"  X: {forces[0]:.4f}")
            print(f"  Y: {forces[1]:.4f}")
            print(f"  Z: {forces[2]:.4f}")
            print(f"Body Moments:")
            print(f"  X: {moments[0]:.4f}")
            print(f"  Y: {moments[1]:.4f}")
            print(f"  Z: {moments[2]:.4f}")

            print(f"L/D {forces[2] / forces[0]:.4f}")
