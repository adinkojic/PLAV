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
from plav.trim_solver import trim_glider_hddot0

import plav.conversions as conv

class WindTunnel(object):
    """Wind Tunnel, for testing flight dynamics models"""
    def __init__(self, scenario_file):
        self.scenario_file = scenario_file
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

        self.pilot_rudder = 0.0
        self.pilot_aileron = 0.0
        self.pilot_elevator = 0.0
        self.pilot_throttle = 0.0

    def reload_vehicle(self):
        """Reload the vehicle model"""
        modelparam = load_scenario("scenarios/" + self.scenario_file)
        self.model, _ = load_aircraft_config(modelparam)

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

    def change_rudder(self, rudder):
        """Model rudder [rad]"""
        self.pilot_rudder = rudder

    def change_aileron(self, aileron):
        """Model aileron [rad]"""
        self.pilot_aileron = aileron

    def change_elevator(self, elevator):
        """Model elevator [rad]"""
        self.pilot_elevator = elevator

    def change_throttle(self, throttle):
        """Model throttle [nd]"""
        self.pilot_throttle = throttle

    def zero_trim(self):
        """Zero out model trim"""
        self.model.update_trim(0.0, 0.0, 0.0, 0.0)

    def trim_out(self, apply = True):
        """Solves the model trim"""
        trim_result = trim_glider_hddot0(
            airspeed=self.airspeed,
            altitude=self.altitude,
            model=self.model
        )

        self.alpha = trim_result['alpha_deg']/conv.RAD_TO_DEG # rad
        self.beta = trim_result['beta_deg']/conv.RAD_TO_DEG # rad

        print(trim_result)

    def update_qbar(self):
        """Called internallly"""
        self.qbar = 0.5 * self.density * self.airspeed**2

    def solve_forces(self, quiet = False):
        """Solve and print forces"""
        wind_velocity = alpha_beta_to_velocity(self.airspeed, self.alpha, self.beta)
        self.model.update_conditions(self.altitude, wind_velocity, self.body_rate, \
                                    self.density, self.pdt[2], self.speed_of_sound)
        self.model.update_control(
            self.pilot_rudder,
            self.pilot_aileron,
            self.pilot_elevator,
            self.pilot_throttle
        )
        forces, moments = self.model.get_forces()

        
        wind_to_body_axis = get_wind_to_body_axis(self.alpha, self.beta)
        forces = rotateFrameQ(wind_to_body_axis, forces)
        coeff_forces = forces / self.qbar / self.S
        
        coeff = self.model.get_coeff()
        control_vec = self.model.get_control_deflection()
        vehicle_rudder = control_vec[0]
        vehicle_aileron = control_vec[1]
        vehicle_elevator = control_vec[2]
        vehicle_throttle = control_vec[3]

        if not quiet:
            print(f"Wind Velocity: {wind_velocity}")
            print("Forces:", forces)

            print("Sref:", self.S)
            print("cref:", self.cbar)
            print("bref:", self.b)

            print(f"Results at {self.airspeed} m/s,"
                f" alpha={self.alpha * conv.RAD_TO_DEG} deg,"
                f" beta={self.beta * conv.RAD_TO_DEG} deg,"
                f" altitude={self.altitude} ft"
                f" rudder={vehicle_rudder:.2f} rad"
                f" aileron={vehicle_aileron:.2f}"
                f" elevator={vehicle_elevator:.2f}"
                f" throttle={vehicle_throttle:.2f}"
            f"\n  Lift Coefficient: {-coeff_forces[2]:.6f}"
            f"\n  Drag Coefficient: {-coeff_forces[0]:.6f}"
            f"\n  Side Force Coefficient: {coeff_forces[1]:.6f}"
            f"\n  Roll Moment Coefficient: {moments[0] / self.qbar / self.b:.6f}"
            f"\n  Pitch Moment Coefficient: {moments[1] / self.qbar / self.cbar:.6f}"
            f"\n  Yaw Moment Coefficient: {moments[2] / self.qbar / self.b:.6f}")

            print(f"Body Forces:")
            print(f"  X: {forces[0]:.6f}")
            print(f"  Y: {forces[1]:.6f}")
            print(f"  Z: {forces[2]:.6f}")
            print(f"Body Moments:")
            print(f"  X: {moments[0]:.6f}")
            print(f"  Y: {moments[1]:.6f}")
            print(f"  Z: {moments[2]:.6f}")

            print(f"L/D {forces[2] / forces[0]:.6f}")
