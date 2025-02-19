"""Jitted Implementation of USSA1976
https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf
"""

import numpy as np
from numba import jit

@jit
def get_pressure_density_temp(altitude):
    """Gets pressure from altitude, page 11 of document

    Parameters
    altitude

    Returns
    pressure, density, temperature array
    """

    start_heights  = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
    start_pressures= np.array([101325, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642 ])
    start_temp = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65 ])
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

@jit
def get_dynamic_viscosity(temperature):
    """Equation 51 of USSA1976"""
    beta = 1.458e-6
    sutherlands = 110.4

    mu = (beta * temperature**1.5)/(temperature+sutherlands)

    return mu
