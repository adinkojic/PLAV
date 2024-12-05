import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84
import brgr_aero_forces_linearized as aero

@jit
def yer(aa):
    bruh = np.array([aa]).clip(-5,5)
    return bruh[0]
velocity = aero.from_alpha_beta(15, 5, 0)

test_state = np.array([0, 0, 0, velocity[0], velocity[1] , velocity[2], 0, 0 ,0, 0, 0, 0, 0])

print(aero.get_aero_forces(test_state))
