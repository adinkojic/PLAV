import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.integrate
import time
import math
from numba import jit
'''
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep 

'''

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

def get_forces(state):
    #TODO: implement a proper gravity model
    g = 9.81
    gravity = np.array([0, 0, -g])
    orientation = np.array([state[6], state[7], state[8], state[9]])
    grotated = rotateFrameQ(orientation, gravity)

    result = grotated
    return result




# x_dot = f(x,u)
# this is the f(x,u)
# returns [x,y,z (NED) , vx, vy, vz, q1 q2 q3 q4, p, q, r]
# TODO: Change to numpyarray
def x_dot(t, y):
    

    #TODO: get real glider values
    mass = 1
    interia = np.identity(3)

    x_dot = np.zeros(13)

    #xdot ydot zdot are vx, vy, vz
    #TODO: convert this from inertial velocity to body velocity
    v_body = np.array([ y[3], y[4], y[5] ])
    orientation = np.array([y[6], y[7], y[8], y[9]])
    v_inertial = rotateVectorQ(orientation, v_body)
    omega = np.array([ y[10], y[11], y[12]])

    x_dot[0] = v_inertial[0]
    x_dot[1] = v_inertial[1]
    x_dot[2] = v_inertial[2]

    #solving for acceleration, which is velocity_dot
    force_body = get_forces(state = y)
    

    v_b_dot = force_body/mass + np.cross(omega, v_body) #this cross term might be broken
    x_dot[3] = v_b_dot[0]
    x_dot[4] = v_b_dot[1]
    x_dot[5] = v_b_dot[2]

    # integrating roll rates to quaternion
    #  r*q2 -q*q3 +p*q4      *0.5 everything
    # -r*q1 +p*q3 +q*q4
    #  q*q1 -p*q2 +r*q4
    # -p*q1 -q*q2 -r*q3
    q1dot = 0.5*( y[12]*y[7] - y[11]*y[8] + y[10]*y[9] )
    q2dot = 0.5*(-y[12]*y[6] + y[10]*y[8] + y[11]*y[9] )
    q3dot = 0.5*( y[11]*y[6] - y[10]*y[7] + y[12]*y[9] )
    q4dot = 0.5*(-y[10]*y[6] - y[11]*y[7] - y[12]*y[8] )
    x_dot[6] = q1dot
    x_dot[7] = q2dot
    x_dot[8] = q3dot
    x_dot[9] = q4dot

    #to solve for angular velocity change
    #w_B = II_B^-1( M_B - w_Bconj * II_B * w_B )
    #moments_body = get_moments(t)
    # TODO: this doesnt work need to fix 
    moments_body = np.zeros(3)
    omega_dot = np.linalg.inv(interia) * (moments_body - np.cross(omega, interia*omega)) #check that matrix inv
    #above is probbably wrong due to issues dimensions and stuff, likely needs a transpose
    x_dot[10] = omega_dot[0][0]
    x_dot[11] = omega_dot[1][0]
    x_dot[12] = omega_dot[2][0]

    return x_dot


start_time = time.time()

int_pos = np.array([1, 0, 15])
int_vel = np.array([0, 1, 0])

angle = -math.pi/2 * 1/2
axis = [0, 1, 0]

int_ori = np.array([math.cos(angle), math.sin(angle)*axis[0], math.sin(angle)*axis[1], math.sin(angle)*axis[2]])
int_rte = np.array([0, 0.5*math.pi, 0])

y0 = np.append(int_pos, np.append(int_vel, np.append(int_ori, int_rte) ))


results = scipy.integrate.solve_ivp(fun = x_dot, t_span = [0, 1], y0=y0, max_step = 0.01)


print("took ", time.time()-start_time)
plt.plot(results.t, results.y[0])
plt.plot(results.t, results.y[1])
plt.plot(results.t, results.y[2])

plt.plot(results.t, results.y[3])
plt.plot(results.t, results.y[4])
plt.plot(results.t, results.y[5])

print(results.y[6][results.t.size-1])
print(results.y[7][results.t.size-1])
print(results.y[8][results.t.size-1])
print(results.y[9][results.t.size-1])
#theta = math.acos(results.y[6])
#plt.plot(results.t, theta)

plt.legend(['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'theta'])
plt.show()