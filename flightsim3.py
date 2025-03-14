"""
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep
This one uses WGS84 and keeps tract of long lat in NED

"""

import json
import time
import math
import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
from numba import jit
import quaternion_math as quat
import brgr_aero_forces_linearized as aero #remove this line
from aircraftconfig import AircraftConfig
#import ussa1976
from atmosphere import Atmosphere

def init_aircraft(config_file):
    """Init aircraft from json file"""
    mass = config_file['mass']
    inertia = np.array(config_file['inertiatensor'])
    cmac = config_file['cref']
    Sref = config_file['Sref']
    bref = config_file['bref']
    C_L0 = config_file['C_L0']
    C_La = config_file['C_La']
    C_D0 = config_file['C_D0']
    epsilon = config_file['k2']
    C_m0 = config_file['C_m0']
    C_ma = config_file['C_ma']
    C_mq = config_file['C_mq']

    C_Y  = config_file['C_Y']
    C_l  = config_file['C_l']
    C_lp = config_file['C_lp']
    C_lr = config_file['C_lr']
    C_np = config_file['C_np']
    C_nr = config_file['C_nr']

    aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, C_L0, C_La, C_D0, epsilon, C_m0, C_ma, C_mq,\
                 C_Y, C_l, C_lp, C_lr, C_np, C_nr )

    return aircraft_model

@jit
def get_gravity(phi, h):
    """gets gravity accel from lat and altitude
    phi: latitude
    h: altitude"""
    graivty = 9.780327*(1 +5.3024e-3*np.sin(phi)**2 - 5.8e-6*np.sin(2*phi)**2) \
            - (3.0877e-6 - 4.4e-9*np.sin(phi)**2)*h + 7.2e-14*h**2
    return graivty

@jit
def x_dot(t, y, aircraft_config, atmosphere):
    """Implements standard NED equations
    [q1 q2 q3 q4], [p q r], (lambda) long, (phi)lat, alt, vn, ve, vd,
    q4 is the angle q13 is the axis """

    x_dot = np.zeros(13)
    q = np.array([y[0], y[1], y[2], y[3]])

    omega = np.array([ y[4], y[5], y[6]])

    long = y[7]
    lat = y[8]
    altitude = y[9]
    vn = y[10]
    ve = y[11]
    vd = y[12]

    mass = aircraft_config.get_mass()
    inertia_tensor = aircraft_config.get_inertia_matrix()



    a = 6378137.0 #earth semi-major axis
    e = 0.0818 #earth ecentricity
    omega_e = 7.292115e-5 #earth rotation rate
    R_phi = a*(1-e**2)/((1-e**2*np.sin(lat)))**(3/2)
    R_lamb = a/math.sqrt((1-e**2*np.sin(lat)))

    omega_NI = omega_e * np.array([np.cos(lat), 0, -np.sin(lat)]) + np.array( \
        [(ve)/(R_lamb + altitude), -(vn)/(R_phi + altitude), -(ve * np.tan(lat))/(R_lamb + altitude)])

    gravity = get_gravity(lat, altitude)

    

    atmosphere.update_conditions(altitude, time = t)

    air_density = atmosphere.get_density()
    air_temperature = atmosphere.get_temperature()

    #adds wind
    v_airspeed = quat.rotateVectorQ(q, np.array([vn, ve, vd]) + atmosphere.get_wind_ned())
    #solving for acceleration, which is velocity_dot
    aircraft_config.update_conditions(altitude,  v_airspeed, omega, air_density, air_temperature)


    body_forces_stab, moments = aircraft_config.get_forces()
    forces_ned = quat.rotateFrameQ(q, body_forces_stab)

    ## figure out forces here
    #in NED frame
    accel_north = forces_ned[0]/mass
    accel_east  = forces_ned[1]/mass
    accel_down  = forces_ned[2]/mass

    omega = omega - quat.rotateFrameQ(q, omega_NI)

    #integrate state
    q1dot = 0.5*(-omega[0]*q[1] -omega[1]*q[2] -omega[2]*q[3])
    q2dot = 0.5*( omega[0]*q[0] +omega[2]*q[2] -omega[1]*q[3])
    q3dot = 0.5*( omega[1]*q[0] -omega[2]*q[1] +omega[0]*q[3])
    q4dot = 0.5*( omega[2]*q[0] +omega[1]*q[1] -omega[0]*q[2])

    #(11.27) in Engineeering Dyanmics (Kasdin and Paley)
    omega_dot = np.linalg.solve(inertia_tensor, moments - np.cross(np.eye(3), omega) @ inertia_tensor @ omega)
    
    lat_dot = vn/(R_phi+altitude)
    long_dot = ve/((R_lamb+altitude)*np.cos(lat))
    altitude_dot = -vd

    #from book Optimal Estimation of Dynamic Systems
    vn_dot = -(long_dot + 2*omega_e)*ve*np.sin(lat) + vn*vd/(R_phi+altitude) + accel_north
    ve_dot = -(long_dot + 2*omega_e)*vn*np.sin(lat) + ve*vd/(R_phi+altitude) + 2*omega_e*vd*np.cos(lat)+accel_east
    vd_dot = -ve**2/(R_lamb+altitude)-vn**2/(R_phi+altitude) - 2*omega_e*ve*np.cos(lat) + gravity + accel_down



    x_dot[0] = q1dot
    x_dot[1] = q2dot
    x_dot[2] = q3dot
    x_dot[3] = q4dot

    x_dot[4] = omega_dot[0]
    x_dot[5] = omega_dot[1]
    x_dot[6] = omega_dot[2]

    x_dot[7] = lat_dot
    x_dot[8] = long_dot
    x_dot[9] = altitude_dot

    x_dot[10] = vn_dot
    x_dot[11] = ve_dot
    x_dot[12] = vd_dot

    return x_dot

def init_state(lat, lon, alt, velocity, bearing, elevation, roll, init_omega):

    init_pos = np.array([lat,lon,alt])

    init_vel = velocity

    #first apply bearing stuff
    init_ori_ned = quat.from_euler(roll*math.pi/180,-elevation*math.pi/180,-bearing*math.pi/180) #roll pitch yaw


    y0 = np.append(np.append(init_ori_ned, init_omega), np.append(init_pos, init_vel))
    return y0


@jit
def q1totheta(q1):
    result = np.zeros(q1.size)
    for i in range(0, q1.size):
        result[i] = np.acos(q1[i])*2 - math.pi
    return result

@jit
def quat_mag(quat, size):
    result = np.zeros(size)
    for i in range(0, size):
        result[i] = np.sqrt(quat[0][i]**2 + quat[1][i]**2 + quat[2][i]**2 + quat[3][i]**2)
    return result

@jit
def quat_euler_helper(q0, q1, q2, q3, size):

    rpy = np.zeros((size, 3))
    for i in range(size):
        rpy[i][:] = quat.to_euler(np.array([q0[i], q1[i], q2[i], q3[i]]))
    return rpy


code_start_time = time.perf_counter()

#load aircraft config
with open('aircraftConfigs/sphere.json', 'r') as file:
    modelparam = json.load(file)
file.close()

aircraft = init_aircraft(modelparam)

wind_alt_profile = np.array([0, 10000], dtype='d')
wind_speed_profile = np.array([6.096, 6.096], dtype='d')
wind_direction_profile = np.array([0, 0], dtype='d')
#init atmosphere config
atmosphere = Atmosphere(wind_alt_profile,wind_speed_profile,wind_direction_profile)


inital_alt = 9144
init_x = 0
init_y = 0


init_airspeed = 0 #meters per second
init_alpha = 0 #degrees
init_beta  = 0
init_velocity = aero.from_alpha_beta(init_airspeed, init_alpha, init_beta)
#init_velocity = np.array([1, 0, 0])


init_rte = np.array([0.0, 0.0, 0.0], dtype='d')

y0 = init_state(init_x, init_y, inital_alt, init_velocity, bearing=0, elevation=0, roll=0, init_omega=init_rte)

#pump sim ocne
scipy.integrate.solve_ivp(fun = x_dot, t_span=[0, 0.001], args= (aircraft,atmosphere), y0=y0, max_step=0.001)

print('Sim started...')

sim_start_time = time.perf_counter()

results=scipy.integrate.solve_ivp(fun=x_dot, t_span=[0, 30], args=(aircraft,atmosphere), y0=y0,max_step=0.001)

sim_end_time = time.perf_counter()



figure, axis = plt.subplots(4,2)

axis[0][0].plot(results.t, results.y[7])
axis[0][0].plot(results.t, results.y[8])

axis[0][1].plot(results.t, results.y[9])

axis[1][1].plot(results.t, results.y[10])
axis[1][1].plot(results.t, results.y[11])
axis[1][1].plot(results.t, results.y[12])
axis[1][0].plot(results.t, results.y[4])
axis[1][0].plot(results.t, results.y[5])
axis[1][0].plot(results.t, results.y[6])

axis[2][0].plot(results.t, results.y[0])
axis[2][0].plot(results.t, results.y[1])
axis[2][0].plot(results.t, results.y[2])
axis[2][0].plot(results.t, results.y[3])

thetas = q1totheta(results.y[3])
axis[3][0].plot(results.t, thetas)

rollpitchyaw=quat_euler_helper(results.y[0], results.y[1], results.y[2],results.y[3],results.t.size)
axis[3][1].plot(results.t, rollpitchyaw)

#omega_dot = omega_dot_helper(results.y[4], results.y[5], results.y[6], results.t.size)
#axis[2][1].plot(results.t, omega_dot)

axis[0][0].legend(['lat', 'lon'])
axis[0][1].legend(['alt'])
axis[1][0].legend(['p', 'q', 'r'])
axis[1][1].legend(['v_x', 'v_y', 'v_z', 'p', 'q', 'r'])
axis[2][0].legend(['q1', 'q2', 'q3', 'q4'])
axis[3][0].legend(['theta'])
axis[3][1].legend(['roll', 'pitch', 'yaw'])

print("code took ", time.perf_counter()-code_start_time)
print("sim took ", sim_end_time-sim_start_time)

print(results.y[1].size)
print(results.y[9][-1]*3.281)

plt.show()
