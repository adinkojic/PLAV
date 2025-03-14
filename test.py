import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84
import brgr_aero_forces_linearized as aero

from aircraftconfig import AircraftConfig

inertiatensor =[
        [0.00256821747 , 0.0, 0.0],
        [0.0, 0.00842101104, 0.0],
        [0.0, 0.0, 0.00975465594]
    ]

Sref = 0.02064491355
cmac = 0.203201016
bref = 0.101598984

mass = 2.2679619056149
omega_deg = [-4, 3, 22]
qbar = 
omega = omega_deg/57.296

C_m = C_mq * q
C_l = C_lp * p #roll TODO:beta dependence
C_n = C_nr * r #yaw force, TODO: beta dependence

body_pitching_moment = C_m * qbar * Sref * cmac
body_yawing_moment   = C_n * qbar * Sref * bref
body_rolling_moment  = C_l * qbar * Sref * bref

moments = 


omega_dot = np.linalg.solve(inertiatensor, moments - np.cross(np.eye(3), omega) @ inertiatensor @ omega)


