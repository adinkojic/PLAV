import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.integrate
import time
import math
import j2_harmonic_gravity as j2grav
from numba import jit
import wgs84
import quaternion_math as quat
'''
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep 

'''


@jit
def get_forces(state):
     # Extract position
    position = np.array([state[0], state[1], state[2],] )

    # Calculate gravitational force using J2 model
    gravity = j2grav.get_gravitational_force(position)

    # Rotate the gravitational force to the body frame
    orientation = np.array([state[6], state[7], state[8], state[9] ])  # Quaternion (q0, q1, q2, q3)
    grotated = quat.rotateFrameQ(orientation, gravity)

    return grotated


# x_dot = f(x,u)
# this is the f(x,u)
# returns [x,y,z (NED) , vx, vy, vz, q1 q2 q3 q4, p, q, r]
@jit
def x_dot(t, y):
    

    #TODO: get real glider values
    mass = 1
    interia = np.identity(3)

    x_dot = np.zeros(13)

    #xdot ydot zdot are vx, vy, vz
    v_body = np.array([ y[3], y[4], y[5] ])
    orientation = np.array([y[6], y[7], y[8], y[9]])
    v_inertial = quat.rotateVectorQ(orientation, v_body)
    omega = np.array([ y[10], y[11], y[12]])

    x_dot[0] = v_inertial[0]
    x_dot[1] = v_inertial[1]
    x_dot[2] = v_inertial[2]

    #solving for acceleration, which is velocity_dot
    force_body = get_forces(y)

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
    moments_body = np.zeros(3)
    omega_dot = np.linalg.inv(interia) * (moments_body - np.cross(omega, interia*omega)) #check that matrix inv
    #above is probbably wrong due to issues dimensions and stuff, likely needs a transpose
    x_dot[10] = omega_dot[0][0]
    x_dot[11] = omega_dot[1][0]
    x_dot[12] = omega_dot[2][0]

    return x_dot

@jit
def q1totheta(q1):
    result = np.zeros(q1.size)
    for i in range(0, q1.size):
        result[i] = math.acos(q1[i])*2
    return result



def init_state(lat, long, alt, velocity, bearing, elevation, roll):
    init_pos = wgs84.from_lat_long_alt(lat, long, alt)
    
    init_vel = velocity

    #first apply bearing stuff
    init_ori_ned = quat.from_euler(roll,elevation,bearing) #roll pitch yaw

    ned_to_wgs84 = wgs84.from_NED_lat_long_h(np.array([lat, long, h]))

    init_ori = quat.mulitply(ned_to_wgs84, init_ori_ned)

    init_rte = np.array([0, 0, 0])

    y0 = np.append(init_pos, np.append(init_vel, np.append(init_ori, init_rte) ))
    return y0

code_start_time = time.time()

inital_alt = 5
inital_lat = 0
inital_lon = 0

init_velocity = np.array([0, 0, 0])

axis = np.array([1, 0, 0])
angle = 0

y0 = init_state(inital_lat, inital_lon, inital_alt, init_velocity, 0, 0, 0)


results = scipy.integrate.solve_ivp(fun = x_dot, t_span = [0, 1], y0=y0, max_step = 0.01)


print("took ", time.time()-code_start_time)

figure, axis = plt.subplots(2,1)

lat_long_h_results = wgs84.to_lat_long_xyz_array(results.y[0:3])

axis[1].plot(results.t, lat_long_h_results[0])
axis[1].plot(results.t, lat_long_h_results[1])
axis[0].plot(results.t, lat_long_h_results[2])

axis[0].plot(results.t, results.y[3])
axis[0].plot(results.t, results.y[4])
axis[0].plot(results.t, results.y[5])

thetas = q1totheta(results.y[6])
axis[0].plot(results.t, thetas)

axis[1].legend(['lat', 'long'])
axis[0].legend(['h', 'v_x', 'v_y', 'v_z', 'theta'])
plt.show()