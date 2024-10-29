import numpy as np
from numba import jit
import math

#frame rotation algorithm
@jit
def rotateFrameQ(quat, vec):
    s = quat[0] #scalar part (rotation angle)
    r = np.array([ quat[1],  quat[2],  quat[3]]) #vector part (rotation axis)
    m = quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]
    
    v_prime = vec + 2 * np.cross(r, (s*vec + np.cross(r, vec)) ) / m
    return v_prime

#vector rotation algorithm
@jit
def rotateVectorQ(quat, vec):
    s = quat[0] #scalar part (rotation angle)
    r = np.array([-quat[1], -quat[2], -quat[3]]) #vector part (rotation axis)
    m = quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]
    
    v_prime = vec + 2 * np.cross(r, (s*vec + np.cross(r, vec)) ) / m
    return v_prime

vector = np.array([1, 0, 0])

angle = math.pi/2
axis = [0, 1, 0]
int_ori = np.array([math.cos(angle * -1/2), math.sin(angle * -1/2)*axis[0], math.sin(angle * -1/2)*axis[1], math.sin(angle * -1/2)*axis[2]])

result_vector = rotateFrameQ(int_ori, vector)
print(result_vector)