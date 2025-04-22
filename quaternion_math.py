"""My own quaternion math library cuz the other one was annoying"""

import math
import numpy as np
from numba import jit, float64, int64

@jit(float64[:](float64[:], float64[:]))
def rotateFrameQ(quat, vec):
    """frame rotation algorithm
    q0 is scalar part"""
    s = quat[0] #scalar part (rotation angle)
    r = np.array([ quat[1],  quat[2],  quat[3]]) #vector part (rotation axis)
    m = quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]

    v_prime = vec + 2 * np.cross(r, (s*vec + np.cross(r, vec)) ) / m
    return v_prime


@jit(float64[:](float64[:], float64[:]))
def rotateVectorQ(quat, vec):
    """Vector rotation algorithm
    q0 is scalar part"""
    s = quat[0] #scalar part (rotation angle)
    r = np.array([-quat[1], -quat[2], -quat[3]], 'd') #vector part (rotation axis)
    m = quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]

    v_prime = vec + 2 * np.cross(r, (s*vec + np.cross(r, vec)) ) / m
    return v_prime


@jit(float64[:](float64, float64[:]))
def from_angle_axis(angle, axis):
    """returns a quaternion given an angle [rad] and axis"""
    return np.array([math.cos(angle/2), math.sin(angle/2)*axis[0], math.sin(angle/2)*axis[1], math.sin(angle/2)*axis[2]])


@jit(float64[:](float64[:], float64[:]))
def mulitply(q1, q2):
    """Multiplies quaternions"""
    q3 = np.zeros(4)
    q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q3[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q3[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return q3

@jit(float64[:](float64[:]))
def to_euler(q):
    """roll, pitch, yaw"""
    phi   = math.atan2(2*(q[0]*q[1] + q[2]*q[3]), (1-2*(q[1]**2 + q[2]**2)))
    theta = 2*math.atan2(math.sqrt(1+2*(q[0]*q[2] - q[1]*q[3])), math.sqrt(1-2*(q[0]*q[2] -q[1]*q[3]))) - math.pi/2
    psi   = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), (1-2*(q[2]**2 + q[3]**2)))

    return np.array([phi, theta, psi]) #idk why you gotta sub pi

@jit(float64[:](float64, float64, float64))
def from_euler(roll, pitch, yaw):
    """gets the rotation quaternion from euler angles [rad]"""
    quat_yaw   = from_angle_axis(yaw,   np.array([0, 0, 1], 'd'))
    quat_pitch = from_angle_axis(pitch, np.array([0, 1, 0], 'd'))
    quat_roll  = from_angle_axis(roll,  np.array([1, 0, 0], 'd'))

    return mulitply(quat_yaw, mulitply(quat_pitch,quat_roll))

@jit(float64[:](float64[:]))
def q1totheta(q1):
    """q1 to theta value around the vector
    assumes q1 is angle part"""
    result = np.zeros(q1.size)
    for i in range(0, q1.size):
        result[i] = np.acos(q1[i])*2 - math.pi
    return result

@jit(float64[:](float64[:,:], int64))
def quat_mag(quat, size):
    """quaternion magnitute, should always be one"""
    result = np.zeros(size)
    for i in range(0, size):
        result[i] = np.sqrt(quat[0][i]**2 + quat[1][i]**2 + quat[2][i]**2 + quat[3][i]**2)
    return result

@jit(float64[:,:](float64[:], float64[:], float64[:], float64[:], int64))
def quat_euler_helper(q0, q1, q2, q3, size):
    """for arrays"""
    rpy = np.zeros((size, 3))
    for i in range(size):
        rpy[i][:] = to_euler(np.array([q0[i], q1[i], q2[i], q3[i]]))
    return rpy
