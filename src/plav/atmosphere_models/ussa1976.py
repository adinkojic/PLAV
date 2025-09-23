"""Atmosphere/enviroment object, used to simulate a 
dynamic enviroment with wind
Implements only USSA1976 for now

https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf"""

import numpy as np
from numba import jit, float64    # import the types
from numba.experimental import jitclass


@jit(float64[:](float64), cache=True)
def get_pressure_density_temp(altitude):
    """Gets pressure from altitude, page 11 of document

    Parameters
    altitude

    Returns
    pressure, density, temperature array
    """

    start_heights  = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
    start_pressures= np.array([101325, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642 ])
    start_temp = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
    temp_lapse = np.array([-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002 ])
    g = 9.80665
    Rs = 287.052874

    #probably should implement this
    if altitude < 0:
        altitude = 0


    #find constant subscript
    b = 0
    while altitude >= start_heights[b]:     #will raise an exeception if alt>84852
        b = b + 1
    b = b - 1
    #two equations depending on if the temperature changes with altitude, found on page 11
    if temp_lapse[b] == 0:
        pressure = start_pressures[b] * np.exp(-(altitude-start_heights[b]) * g / Rs /start_temp[b])
        temperature = start_temp[b]
        density = pressure / Rs / temperature
    else:
        pressure = start_pressures[b] * \
              ( start_temp[b]/(start_temp[b] + temp_lapse[b] * (altitude-start_heights[b])) ) \
                **(g / Rs / temp_lapse[b])
        temperature = start_temp[b] + temp_lapse[b] * (altitude-start_heights[b])
        density = pressure / Rs / temperature

    return np.array([pressure, density, temperature])

@jit(float64(float64), cache=True)
def get_speed_of_sound(temperature_K):
    """Speed of sounds [m/s]"""
    gamma_air = 1.4
    r_air = 287.052874

    return np.sqrt(gamma_air*r_air*temperature_K)

@jit(float64(float64), cache=True)
def get_dynamic_viscosity(temperature):
    """Equation 51 of USSA1976"""
    beta = 1.458e-6
    sutherlands = 110.4

    mu = (beta * temperature**1.5)/(temperature+sutherlands)

    return mu


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

    def __init__(self, wind_alt_profile = np.array([0.0,0.0]),
                 wind_speed_profile = np.array([0.0,0.0]),
                 wind_direction_profile = np.array([0.0,0.0])):
        self.wind_alt       = wind_alt_profile
        self.wind_speed     = wind_speed_profile
        self.wind_direction = wind_direction_profile

        self.current_alt = 0.0
        self.current_time = 0.0
        self.current_pressure = 0.0
        self.current_density = 0.0
        self.current_temperature = 0.0

    def change_wind_profile(self, wind_alt_profile, wind_speed_profile, wind_direction_profile):
        """updates the wind profile"""
        self.wind_alt       = wind_alt_profile
        self.wind_speed     = wind_speed_profile
        self.wind_direction = wind_direction_profile

    def update_conditions(self, altitude, time = 0.0):
        """Updates experienced atmosphere conditons
        must be called every timestep"""
        self.current_alt = altitude
        self.current_time = time

        pdt = get_pressure_density_temp(altitude)
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
        """Returns speed of sound [m/s]"""
        return get_speed_of_sound(self.current_temperature)

    def get_wind_ned(self):
        """Returns wind speed NED [m/s]"""
        wind_speed = np.interp(self.current_alt, self.wind_alt, self.wind_speed)
        wind_direction = np.interp(self.current_alt, self.wind_alt, self.wind_direction)

        wind_east  = np.sin(wind_direction *0.017453292519943295) * wind_speed
        wind_north = np.cos(wind_direction *0.017453292519943295) * wind_speed
        wind_down = 0.0

        return np.array([wind_north, wind_east, wind_down], 'd')
