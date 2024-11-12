import numpy as np
import math
from numba import jit

#frame rotation algorithm
@jit #this line makes it use numba's compiler that runs much faster (closer to C!)
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

@jit
def from_angle_axis(angle, axis):
    return np.array([math.cos(angle/2), math.sin(angle/2)*axis[0], math.sin(angle/2)*axis[1], math.sin(angle/2)*axis[2]])

#multiplies quaternions
@jit(parallel = True)
def mulitply(q1, q2):
    q3 = np.zeros(4)
    q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q3[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q3[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return q3

#aka roll, pitch, yaw
#@jit
def to_euler(q):
    phi   = math.atan2(2*(q[0]*q[1] + q[2]*q[3]), (1-2*(q[1]**2 + q[2]**2)))
    theta = 2*math.atan2(math.sqrt(1+2*(q[0]*q[2] - q[1]*q[3])), math.sqrt(1-2*(q[0]*q[2] -q[1]*q[3]))) - math.pi/2
    psi   = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), (1-2*(q[2]**2 + q[3]**2)))

    return np.array([phi, theta, psi]) #idk why you gotta sub pi

#@jit
def from_euler(roll, pitch, yaw):
    quat_yaw   = from_angle_axis(yaw,   np.array([0, 0, 1]))
    quat_pitch = from_angle_axis(pitch, np.array([0, -1, 0]))
    quat_roll  = from_angle_axis(roll,  np.array([-1, 0, 0]))

    return mulitply(quat_roll, mulitply(quat_pitch,quat_yaw))
