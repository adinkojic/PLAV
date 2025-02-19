"""
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep 

"""

import json
import time
import math
import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
from numba import jit, float64
import j2_harmonic_gravity as j2grav
import wgs84
import quaternion_math as quat
import brgr_aero_forces_linearized as aero #remove this line
from aircraftconfig import AircraftConfig


def init_aircraft(config_file):
    mass = config_file['mass']
    inertia = np.array(config_file['inertiatensor'])
    cmac = config_file['cref']
    Sref = config_file['Sref']
    bref = config_file['bref']
    C_L0 = config_file['C_L0']
    C_La = config_file['C_La']
    C_D0 = config_file['C_D0']
    epsilon = config_file['k2']
    C_M0 = config_file['C_M0']
    C_Ma = config_file['C_Ma']
    C_Mq = config_file['C_Mq']

    aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, C_L0, C_La, C_D0, epsilon, C_M0, C_Ma, C_Mq)

    return aircraft_model


@jit
def get_forces(state, aircraft_config):
    """gets forces in body frame, due to gravity aero etc"""
     # Extract position
    #position = wgs84.from_lat_long_alt(0,0,state[2]) #this will need to be rotated
    #gravity = j2grav.get_gravitational_force(position)
    gravity = np.array([0, 0, -9.81])*aircraft_config.get_mass()

    #Aero forces in body frame
    aero_forces, aero_moments = aircraft_config.get_forces()

    # Rotate the gravitational force to the body frame
    orientation = np.array([state[6], state[7], state[8], state[9] ])  # Quaternion (q0, q1, q2, q3)
    grotated = quat.rotateVectorQ(orientation, gravity)
    arotated = quat.rotateVectorQ(orientation, aero_forces)

    total_forces = grotated + arotated

    return total_forces

@jit
def x_dot(t, y, aircraft_config):
    """ x_dot = f(x,u)
    this is the f(x,u)
    returns [x,y,z (NED) , vx, vy, vz, q1 q2 q3 q4, p, q, r]"""

    mass = aircraft_config.get_mass()
    interia = aircraft_config.get_inertia_matrix()

    x_dot = np.zeros(13)

    #xdot ydot zdot are vx, vy, vz
    v_body = np.array([ y[3], y[4], y[5] ])
    orientation = np.array([y[6], y[7], y[8], y[9]])
    v_inertial = quat.rotateFrameQ(orientation, v_body)
    omega = np.array([ y[10], y[11], y[12]])

    x_dot[0] = v_inertial[0]
    x_dot[1] = v_inertial[1]
    x_dot[2] = v_inertial[2]

    #solving for acceleration, which is velocity_dot
    aircraft_config.update_conditions(y[2],  v_body, omega )

    force_body = get_forces(y, aircraft_config)

    v_b_dot = force_body/mass - np.cross(omega, v_body)#this cross term might be broken
    x_dot[3] = v_b_dot[0]
    x_dot[4] = v_b_dot[1]
    x_dot[5] = v_b_dot[2]

    # integrating roll rates to quaternion
    #  r*q2 -q*q3 +p*q4      *0.5 everything
    # -r*q1 +p*q3 +q*q4
    #  q*q1 -p*q2 +r*q4
    # -p*q1 -q*q2 -r*q3

    p = y[10]
    q = y[11]
    r = y[12]

    q1 = y[6]
    q2 = y[7]
    q3 = y[8]
    q4 = y[9]

    q1dot = 0.5*(-p*q2 -q*q3 -r*q4)
    q2dot = 0.5*( p*q1 +r*q3 -q*q4)
    q3dot = 0.5*( q*q1 -r*q2 +p*q4)
    q4dot = 0.5*( r*q1 +q*q2 -p*q3)
    x_dot[6] = q1dot
    x_dot[7] = q2dot
    x_dot[8] = q3dot
    x_dot[9] = q4dot

    #to solve for angular velocity change
    #w_B = II_B^-1( M_B - w_Bconj * II_B * w_B )
    #moments_body = get_moments(t)
    moments_body = np.zeros(3)
    omega_dot = np.linalg.inv(interia) * (moments_body - np.cross(omega, interia*omega))*0 #check that matrix inv
    #above is probbably wrong due to issues dimensions and stuff, likely needs a transpose
    x_dot[10] = omega_dot[0][0]
    x_dot[11] = omega_dot[1][0]
    x_dot[12] = omega_dot[2][0]

    return x_dot

@jit(float64(float64[:]))
def q1totheta(q1):
    result = np.zeros(q1.size)
    for i in range(0, q1.size):
        result[i] = np.acos(q1[i])*2
    return result

def from_alpha_beta(airspeed, alpha, beta):
    """gives velocity in body frame from airspeed alpha beta
    not jitted"""
    x = airspeed * math.cos(alpha * math.pi/180) * math.cos(beta * math.pi/180)
    y = airspeed * math.sin(beta  * math.pi/180)
    z = airspeed * math.sin(alpha * math.pi/180) * math.cos(beta * math.pi/180)

    return np.array([x,y,z])

@jit
def velocity_to_alpha_beta(velocity_body):
    """Gets Velocity, sans wind to airspeed, alpha, beta"""
    airspeed = math.sqrt(velocity_body[0]**2 + velocity_body[1]**2 + velocity_body[2]**2)
    temp = math.sqrt(velocity_body[0]**2 + velocity_body[2]**2)
    beta = math.atan2(velocity_body[1], temp)
    alpha = math.atan2(velocity_body[2], velocity_body[0])

    return airspeed, beta, alpha


def init_state(x, y, alt, velocity, bearing, elevation, roll, init_omega):
    #init_pos = wgs84.from_lat_long_alt(lat, long, alt)
    init_pos = np.array([x,y,alt])

    init_vel = velocity

    #first apply bearing stuff
    init_ori_ned = quat.from_euler(roll*math.pi/180,-elevation*math.pi/180,-bearing*math.pi/180) #roll pitch yaw

    #ned_to_wgs84 = wgs84.from_NED_lat_long_h(np.array([lat, long, alt]))

    #init_ori = quat.mulitply(init_ori_ned, ned_to_wgs84)


    y0 = np.append(init_pos, np.append(init_vel, np.append(init_ori_ned, init_omega) ))
    return y0


code_start_time = time.perf_counter()


#load aircraft config
with open('openvspmodel.json', 'r') as file:
    modelparam = json.load(file)
file.close()

#if there's anything to do with mass or date do it now...
del modelparam["modeldate"]
del modelparam["hasgridfins"]
del modelparam["lengthunits"]
del modelparam["angleunits"]
del modelparam["massunits"]

aircraft = init_aircraft(modelparam)


inital_alt = 15
init_x = 0
init_y = 0


init_airspeed = 18 #meters per second
init_alpha = 0 #degrees
init_beta  = 0
init_velocity = aero.from_alpha_beta(init_airspeed, init_alpha, init_beta)
#init_velocity = np.array([1, 0, 0])


init_rte = np.array([0, 0, 0])

y0 = init_state(init_x, init_y, inital_alt, init_velocity, bearing=0, elevation=0, roll=0, init_omega=init_rte)

#pump sim ocne
scipy.integrate.solve_ivp(fun = x_dot, t_span = [0, 0.001], args= (aircraft,), y0=y0, max_step = 0.001)

sim_start_time = time.perf_counter()

results = scipy.integrate.solve_ivp(fun = x_dot, t_span = [0, 30], args= (aircraft,), y0=y0, max_step = 0.001)

sim_end_time = time.perf_counter()

print("code took ", time.perf_counter()-code_start_time)
print("sim took ", sim_end_time-sim_start_time)

figure, axis = plt.subplots(4,1)

axis[0].plot(results.t, results.y[0])
axis[0].plot(results.t, results.y[1])
axis[0].plot(results.t, results.y[2])

axis[1].plot(results.t, results.y[3])
axis[1].plot(results.t, results.y[4])
axis[1].plot(results.t, results.y[5])
axis[1].plot(results.t, results.y[10])
axis[1].plot(results.t, results.y[11])
axis[1].plot(results.t, results.y[12])

axis[2].plot(results.t, results.y[6])
axis[2].plot(results.t, results.y[7])
axis[2].plot(results.t, results.y[8])
axis[2].plot(results.t, results.y[9])

thetas = q1totheta(results.y[6])
axis[0].plot(results.t, thetas)



axis[3] 



axis[0].legend(['x', 'y', 'z', 'theta'])
axis[1].legend(['v_x', 'v_y', 'v_z', 'p', 'q', 'r'])
axis[2].legend(['q1', 'q2', 'q3', 'q4'])
plt.show()
