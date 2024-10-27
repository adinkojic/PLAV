import math
import numpy as np

# Example usage:
def flat_earth_eom(t, x, amod):
    """Funtion flat_earth_eom.py contains the essential elements of a six-degree of freedom
    simulation. The purpose of this function is to allow the numerical approxiamtion of
    solutions of the governing equations for an aircraft.

    The naming convention is <variable name>_<corrdinate system if applicable>_<units>. For
    example, the pitch rate, q resolved in body fixed frame, bf, with units of radians
    per second is named, q_b_rps.

    Arguments:
    t - time [s], scaler
    x - state vector at time t [various units], numpy array
        x[0] - u_b_mps, axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS
        x[1] - v_b_mps, lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS
        x[2] - w_b_mps, vertical velocity of CM wrt inertial CS resolved in aircraft body fixed CS
        x[3] - p_b_rps, roll angular velocity of body fixed CS with respect to inertial CS
        x[4] - q_b_rps, pitch angular velocity of body fixed CS with respect to inertial CS
        x[5] - r_b_rps, yaw angular velocity of body fixed CS with respect to inertial CS
        x[6] - phi_rad, roll angle
        x[7] - theta_rad, pitch angle
        x[8] - psi_rad, yaw angle
        x[9] - p1_n_m, x-axis postition of aircraft resolved in NED CS
        x[10] - p2_n_m, y-axis postition of aircraft resolved in NED CS
        x[11] - p3_n_m, z-axis postition of aircraft resolved in NED CS
    amod - aircraft model data stored as a dictionary containing various parameters

Returns:
    dx - time derivative of each state in x (RHS of governing equations)

History: 
    Original code form Ben Dickinson BIG THANKS TO HIM <3
"""

# Preallocate left hand side of equations
dx = np.array((12,1))

#Assign current state values to variable names
u_b_mps = x[0]
v_b_mps = x[1]
w_b_mps = x[2]
p_b_rps = x[3]
q_b_rps = x[4]
r_b_rps = x[5]
phi_rad = x[6]
theta_rad = x[7]
psi_rad = x[8]
p1_n_m = x[9]
p2_n_m = x[10]
p3_n_m = x[11]

# Get mass and moments of inertia
m_kg = amod([m_kg])
Jxz_b_kgm2 = amod([Jxz_b_kgm2])
Jxx_b_kgm2 = amod([Jxx_b_kgm2])
Jyy_b_kgm2 = amod([Jyy_b_kgm2])
Jzz_b_kgm2 = amod([Jzz_b_kgm2])

# Air data calculation (Mach, altitude, AoA, AoS) (Comming soon)

# Atmosphere model (Coming soon)

# Gravity acts normal to earth tangent CS
gz_n_mps2 = 9.81

# Resolve gravity in body coord system
gx_b_mps2 = -math.sin(theta_rad)*gz_n_mps2
gy_b_mps2 = mathsin(phi_rad)*math.cos(theta_rad)* gz_n_mps2



