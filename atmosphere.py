"""Atmosphere/enviroment object, used to simulate a 
dynamic enviroment with wind
Implements only USSA1976 for now"""

import numpy as np
from numba import float64    # import the types
from numba.experimental import jitclass
from numba import jit

import ussa1976

spec = [
    #wind profile input
    ('wind_alt', float64[:]),
    ('wind_speed', float64[:]),
    ('wind_direction', float64[:]),

    #current values
    ('current_alt', float64),
    ('current_time', float64),
    ('current_density', float64),
    ('current_temperature', float64),
    ('current_pressure', float64)
]

@jitclass(spec)
class Atmosphere(object):
    "Atmosphere jit'd object, storing the wind profile and atmosphere model"

    def __init__(self, wind_alt_profile, wind_speed_profile, wind_direction_profile):
        self.wind_alt       = wind_alt_profile
        self.wind_speed     = wind_speed_profile
        self.wind_direction = wind_direction_profile

        self.current_alt = 0.0
        self.current_time = 0.0
        self.current_pressure = 0.0
        self.current_density = 0.0
        self.current_temperature = 0.0

    def update_conditions(self, altitude, time = 0.0):
        """Updates experienced atmosphere conditons
        must be called every timestep"""
        self.current_alt = altitude
        self.current_time = time

        pdt = ussa1976.get_pressure_density_temp(altitude)
        self.current_pressure    = pdt[0]
        self.current_density     = pdt[1]
        self.current_temperature = pdt[2]

    def get_density(self):
        """Returns density [kg/m^3]"""
        return self.current_density

    def get_temperature(self):
        """Returns temperature [K]"""
        return self.current_temperature

    def get_pressure(self):
        """Returns pressure [Pa]"""
        return self.current_pressure

    def get_speed_of_sound(self):
        """Speed of sounds [m/s]"""
        gamma_air = 1.4
        r_air = 287.052874

        return np.sqrt(gamma_air*r_air*self.current_temperature)

    def get_wind_ned(self):
        """Returns wind speed NED [m/s]"""
        wind_speed = np.interp(self.current_alt, self.wind_alt, self.wind_speed)
        wind_direction = np.interp(self.current_alt, self.wind_alt, self.wind_direction)

        wind_east  = np.sin(wind_direction *0.017453292519943295) * wind_speed
        wind_north = np.cos(wind_direction *0.017453292519943295) * wind_speed
        wind_down = 0

        return np.array([wind_north, wind_east, wind_down])
