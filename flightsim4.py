"""
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep
This one uses WGS84 and keeps tract of long lat in NED

Refactored 3 to OOO style

"""

import json
import time
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import jit
import quaternion_math as quat
import brgr_aero_forces_linearized as aero #remove this line
from aircraftconfig import AircraftConfig, init_aircraft
#import ussa1976
from atmosphere import Atmosphere
from ivp_logger import IVPLogger


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


    body_forces_body, moments = aircraft_config.get_forces()
    forces_ned = quat.rotateFrameQ(q, body_forces_body)

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
    """initalize the state"""


    init_pos = np.array([lat,lon,alt])

    init_vel = velocity

    #first apply bearing stuff
    init_ori_ned = quat.from_euler(roll*math.pi/180,-elevation*math.pi/180,-bearing*math.pi/180) #roll pitch yaw


    y0 = np.append(np.append(init_ori_ned, init_omega), np.append(init_pos, init_vel))
    return y0

code_start_time = time.perf_counter()

#load aircraft config
with open('aircraftConfigs/case3tumblingBrickDamping.json', 'r') as file:
    modelparam = json.load(file)
file.close()

aircraft = init_aircraft(modelparam)

use_file_atmosphere = True
if use_file_atmosphere:
    wind_alt_profile       = np.array(modelparam['wind_alt_profile'], dtype='d')
    wind_speed_profile     = np.array(modelparam['wind_speed_profile'], dtype='d')
    wind_direction_profile = np.array(modelparam['wind_direction_profile'], dtype='d')
else:
    wind_alt_profile = np.array([0, 0], dtype='d')
    wind_speed_profile = np.array([0, 0], dtype='d')
    wind_direction_profile = np.array([0, 0], dtype='d')
#init atmosphere config
atmosphere = Atmosphere(wind_alt_profile,wind_speed_profile,wind_direction_profile)

use_file_init_conditions = True
if use_file_init_conditions:
    inital_alt    = modelparam['init_alt']
    init_velocity = modelparam['init_vel']
    init_rte      = modelparam['init_rot']

    init_x = modelparam['init_lat']
    init_y = modelparam['init_lon']
else:
    inital_alt = 9144
    init_x = 0
    init_y = 0

    init_airspeed = 20 #meters per second
    init_alpha = 0 #degrees
    init_beta  = 0
    #init_velocity = aero.from_alpha_beta(init_airspeed, init_alpha, init_beta)
    init_velocity = [0.0, 0.0, 0.0]
    init_rte = np.array([0.0, 0.0, 0.0], dtype='d')


class Simulator(object):
    """A sim object is required to store all the required data nicely."""
    def __init__(self, init_state, time_span, aircraft, atmosphere, t_step = 0.1):
        self.state = init_state
        self.t_span = time_span
        self.time = time_span[0]
        self.logger = IVPLogger(14)
        self.t_step = t_step
        self.aircraft = aircraft
        self.atmosphere = atmosphere

        self.logger.append_data(np.append(time_span[0], [init_state]))

    def advance_timestep(self):
        """advance timestep function, updates timestep and saves values"""

        local_t_span=np.array([self.time, self.time + self.t_step])
        new_state = solve_ivp(fun = x_dot, t_span=local_t_span, args= (aircraft,atmosphere), y0=self.state)
        self.state = new_state.y[:,-1]
        self.time = new_state.t[-1]

        time = np.array([new_state.t[-1]])
 
        data_to_append = np.append(time, [new_state.y[:,-1]])

        self.logger.append_data(data_to_append)

    def run_sim(self):
        """runs the sim, could also include control inputs"""
        while self.time < self.t_span[1]:
            self.advance_timestep()

    def return_results(self):
        """logger"""
        return self.logger.return_data()
    
    def return_time_steps(self):
        """returns number of timesteps saved"""
        return self.logger.return_data_size()

y0 = init_state(init_x, init_y, inital_alt, init_velocity, bearing=0, elevation=0, roll=0, init_omega=init_rte)

#pump sim ocne
solve_ivp(fun = x_dot, t_span=[0, 0.001], args= (aircraft,atmosphere), y0=y0, max_step=0.001)

t_span = np.array([0.0, 30.0])

sim_object = Simulator(y0, t_span, aircraft, atmosphere, t_step=0.001)

print("Sim started...")

sim_start_time = time.perf_counter()
sim_object.run_sim()
sim_end_time = time.perf_counter()

sim_data = sim_object.return_results()

figure, axis = plt.subplots(4,2)

print(sim_object.return_time_steps())
print(np.size(sim_data[:,10]))

axis[0][0].plot(sim_data[:,0], sim_data[:,8]*180/math.pi)
axis[0][0].plot(sim_data[:,0], sim_data[:,9]*180/math.pi)

axis[0][1].plot(sim_data[:,0], sim_data[:,10]*3.281)

axis[1][1].plot(sim_data[:,0], sim_data[:,11]*3.281)
axis[1][1].plot(sim_data[:,0], sim_data[:,12]*3.281)
axis[1][1].plot(sim_data[:,0], sim_data[:,13]*3.281)
axis[1][0].plot(sim_data[:,0], sim_data[:,5]*180/math.pi)
axis[1][0].plot(sim_data[:,0], sim_data[:,6]*180/math.pi)
axis[1][0].plot(sim_data[:,0], sim_data[:,7]*180/math.pi)

axis[2][0].plot(sim_data[:,0], sim_data[:,1])
axis[2][0].plot(sim_data[:,0], sim_data[:,2])
axis[2][0].plot(sim_data[:,0], sim_data[:,3])
axis[2][0].plot(sim_data[:,0], sim_data[:,0])

thetas = quat.q1totheta(sim_data[:,4])
axis[3][0].plot(sim_data[:,0], thetas)

rollpitchyaw=quat.quat_euler_helper(sim_data[:,1], sim_data[:,2], sim_data[:,3],sim_data[:,4],sim_data[:,0].size)
axis[3][1].plot(sim_data[:,0], rollpitchyaw *180/math.pi)

#omega_dot = omega_dot_helper(results.y[4], results.y[5], results.y[6], results.t.size)
#axis[2][1].plot(results.t, omega_dot)

axis[0][0].legend(['lat', 'lon'])
axis[0][1].legend(['alt'])
axis[1][0].legend(['p', 'q', 'r'])
axis[1][1].legend(['v_n', 'v_e', 'v_d', 'p', 'q', 'r'])
axis[2][0].legend(['q1', 'q2', 'q3', 'q4'])
axis[3][0].legend(['theta'])
axis[3][1].legend(['roll', 'pitch', 'yaw'])

print("code took ", time.perf_counter()-code_start_time)
print("sim took ", sim_end_time-sim_start_time)

print(sim_data[:,0].size)
print(sim_data[-1][10]*3.281)
print(sim_data[-1][9]* 57.296)

plt.show()
