
import math
from numba import jit, float64
import numpy as np

@jit(float64[:,:](float64))
def get_x_rotation_matrix(angle):
    """gets a rotation matrix about X, useful for grid fins [rad]"""
    rotation_around_body = np.array([ [1.0, 0, 0], \
                         [0, np.cos(angle), -np.sin(angle)], \
                         [0, np.sin(angle), np.cos(angle)] ], 'd')
    return rotation_around_body

print(get_x_rotation_matrix(math.pi))
