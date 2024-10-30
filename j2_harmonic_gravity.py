import math
import numpy as np
from numba import jit

@jit
def get_gravitational_force(position):
    # Constants
    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    M = 5.972e24     # Mass of the Earth (kg)
    R = 6371e3       # Radius of the Earth (m)
    J2 = 1.08263e-3  # J2 coefficient for Earth's gravitational potential

    """Calculate gravitational force based on J2 potential."""
    r = np.linalg.norm(position)
    
    # Latitude φ (phi) calculation from position
    latitude = np.arcsin(position[2] / r)

    # Calculate gravitational potential U(r, φ)
    #U = (G * M / r) * (1 - (J2 / 2) * (3 * np.sin(latitude)**2 - 1))

    # Calculate gravitational force from potential
    # Gravitational acceleration g = -∇U
    g_r = G * M / r**2  # Radial component
    g_latitude = (3/2) * J2 * (G * M * R**2) / r**4 * np.sin(latitude) * np.cos(latitude)  # J2 contribution

    # Gravitational force vector
    force = np.array([
        -g_r * (position[0] / r) + g_latitude * (3 * position[0] * position[2]) / (r**2),
        -g_r * (position[1] / r) + g_latitude * (3 * position[1] * position[2]) / (r**2),
        -g_r * (position[2] / r) + g_latitude * (3 * position[2]**2 - r**2) / (r**3)
    ])

    return force
