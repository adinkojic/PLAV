"""
Simulation object lives here, along with the x_dot function

Gravity lives here as well (J2)

"""

import math
import time

import numpy as np
#from scipy.integrate import solve_ivp
from numba import jit, float64

from plav.quaternion_math import rotateFrameQ, rotateVectorQ, to_euler
from plav.step_logging import SimDataLogger
from plav.runge_kutta4 import basic_rk4
from plav.atmosphere_models.ussa1976 import Atmosphere
from plav.vehicle_models.generic_aircraft_config import AircraftConfig

@jit(float64(float64,float64))
def get_gravity(phi, h):
    """gets gravity accel from lat and altitude
    phi: latitude [rad]
    h: altitude [m]
    returns: gravity acceleration [m/s^2]"""
    gravity = 9.780327*(1 + 5.3024e-3*np.sin(phi)**2 - 5.8e-6*np.sin(2*phi)**2) \
            - (3.0877e-6 - 4.4e-9*np.sin(phi)**2)*h + 7.2e-14*h**2
    return gravity

@jit
def x_dot(t, y, aircraft_config: AircraftConfig, sim_atmosphere: Atmosphere, log = None):
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
        [(ve)/(R_lamb + altitude),-(vn)/(R_phi + altitude),-(ve * np.tan(lat))/(R_lamb + altitude)])

    gravity = get_gravity(lat, altitude)

    sim_atmosphere.update_conditions(altitude)

    air_density = sim_atmosphere.get_density()
    air_temperature = sim_atmosphere.get_temperature()
    static_pressure = sim_atmosphere.get_pressure()
    speed_of_sound = sim_atmosphere.get_speed_of_sound()

    #adds wind
    v_airspeed = rotateVectorQ(q, np.array([vn, ve, vd], 'd') + sim_atmosphere.get_wind_ned())
    #solving for acceleration, which is velocity_dot
    aircraft_config.update_conditions(altitude,  v_airspeed, omega, air_density, 
                                      air_temperature, speed_of_sound)


    aero_forces_body, aero_moments = aircraft_config.get_forces()
    aircraft_thrust = aircraft_config.calculate_thrust()

    body_forces_body = np.array([
        aero_forces_body[0] + aircraft_thrust,
        aero_forces_body[1],
        aero_forces_body[2],
        ], 'd')

    accel_body = body_forces_body/mass

    accel_ned = rotateFrameQ(q, accel_body)


    accel_north = accel_ned[0]
    accel_east  = accel_ned[1]
    accel_down  = accel_ned[2]

    omega = omega - rotateFrameQ(q, omega_NI)

    #integrate state
    q1dot = 0.5*(-omega[0]*q[1] -omega[1]*q[2] -omega[2]*q[3])
    q2dot = 0.5*( omega[0]*q[0] +omega[2]*q[2] -omega[1]*q[3])
    q3dot = 0.5*( omega[1]*q[0] -omega[2]*q[1] +omega[0]*q[3])
    q4dot = 0.5*( omega[2]*q[0] +omega[1]*q[1] -omega[0]*q[2])

    #(11.27) in Engineeering Dyanmics (Kasdin and Paley)
    omega_dot = np.linalg.solve(inertia_tensor, aero_moments - \
                                np.cross(np.eye(3), omega) @ inertia_tensor @ omega)

    lat_dot = vn/(R_phi+altitude)
    long_dot = ve/((R_lamb+altitude)*math.cos(lat))
    altitude_dot = -vd

    #from book Optimal Estimation of Dynamic Systems
    vn_dot = accel_north-(long_dot + 2*omega_e)*ve*np.sin(lat) + vn*vd/(R_phi+altitude)
    ve_dot = accel_east -(long_dot + 2*omega_e)*vn*np.sin(lat) + ve*vd/(R_phi+altitude)\
             + 2*omega_e*vd*np.cos(lat)
    vd_dot = accel_down + gravity-ve**2/(R_lamb+altitude)-vn**2/(R_phi+altitude)\
          - 2*omega_e*ve*np.cos(lat)



    x_dot[0] = q1dot
    x_dot[1] = q2dot
    x_dot[2] = q3dot
    x_dot[3] = q4dot

    x_dot[4] = omega_dot[0]
    x_dot[5] = omega_dot[1]
    x_dot[6] = omega_dot[2]

    x_dot[7] = long_dot
    x_dot[8] = lat_dot
    x_dot[9] = altitude_dot

    x_dot[10] = vn_dot
    x_dot[11] = ve_dot
    x_dot[12] = vd_dot

    mach = aircraft_config.get_mach()
    dynamic_pressure = aircraft_config.get_qbar()
    true_airspeed = aircraft_config.get_airspeed()

    alpha = aircraft_config.get_alpha()
    beta  = aircraft_config.get_beta()
    reynolds = aircraft_config.get_reynolds()

    control_deflection = aircraft_config.get_control_deflection()

    if log is not None:
        log.load_line(t, y, aero_forces_body, \
                    aero_moments, gravity, speed_of_sound, mach ,dynamic_pressure, \
                    true_airspeed, air_density, static_pressure, air_temperature, \
                    alpha, beta, reynolds, aircraft_thrust, control_deflection,\
                    accel_body)

    return x_dot

class Simulator(object):
    """A sim object is required to store all the required data nicely."""
    def __init__(self,
            init_state, time_span, aircraft: AircraftConfig, sim_atmosphere: Atmosphere,
            control_sys= None,t_step = 0.1
                ):
        self.state = init_state
        self.t_span = time_span
        self.time = time_span[0]
        self.sim_log = SimDataLogger(preallocated=1.1*(time_span[1]-time_span[0]) / t_step)
        self.t_step = t_step
        self.aircraft = aircraft
        self.sim_atmosphere = sim_atmosphere
        self.start_time = None

        self.paused = True
        self.elapsed_time = 0.0
        self.time_at_last_pause = 0.0

        self.pilot_vec = np.zeros(4, 'd')
        self.control_sys = control_sys

        #log the inital state
        x_dot(self.time, self.state, aircraft, sim_atmosphere, self.sim_log)
        self.sim_log.save_line()

    def advance_timestep(self):
        """advance timestep function, updates timestep and saves values"""

        if self.control_sys is not None:
            #by now the HIL should have a response ready
            #for HIL it might block a bit as the aurdino computes
            total_control_vector = self.control_sys_request_response()
            self.aircraft.update_control(total_control_vector)
        else:
            total_control_vector = self.pilot_vec
            self.aircraft.update_control(total_control_vector)

        self.time, self.state = basic_rk4(x_dot, self.time, self.t_step, self.state,\
                                           args= (self.aircraft,self.sim_atmosphere))

        #lon wrapparound
        if self.state[7] < -math.pi:
            self.state[7] = self.state[7] + 2.0*math.pi
        elif self.state[7] > math.pi:
            self.state[7] = self.state[7] - 2.0*math.pi

        #get stuff
        x_dot(self.time, self.state, self.aircraft, self.sim_atmosphere, self.sim_log)
        self.sim_log.save_line()

        if self.control_sys is not None:
            #tell the control system to update now in case its a HIL system so it has time
            self.control_sys_update()

    def control_sys_update(self):
        """updates the control system with the latest data"""
        last_line = self.sim_log.get_lastest()

        if last_line is not None:

            self.control_sys.update_enviroment(last_line)

            pilot_control_lat = self.pilot_vec[0]
            pilot_control_yaw = self.pilot_vec[1]
            pilot_control_long = self.pilot_vec[2]
            pilot_control_throttle = self.pilot_vec[3]
            self.control_sys.update_pilot_control(pilot_control_long, pilot_control_lat, \
                        pilot_control_yaw, pilot_control_throttle)

    def control_sys_request_response(self):
        """request a reponse from the control system, which should have a response ready"""
        control_vec = self.control_sys.get_control_output()
        return control_vec

    def latest_state(self):
        """returns the most recent state
        in lat [rad], lon [rad], alt [m], psi [rad], theta [rad], phi[rad]"""
        lat_lon_alt = self.state[7:10]
        psi_theta_phi = to_euler(self.state[0:4])

        return np.concatenate((lat_lon_alt,psi_theta_phi))

    def update_real_time(self, time_warp = 1.0):
        """Updates the real time sim, try to call with a delay in between"""
        if self.start_time is None:
            self.start_time = time.time()

        if self.paused:
            pass
        else:
            self.elapsed_time = (time.time() - self.start_time) * time_warp +self.time_at_last_pause
            while self.time < self.elapsed_time:
                self.advance_timestep()
        return self.return_results()

    def change_aircraft(self, new_aircraft_config: AircraftConfig):
        """Changes the aircraft for future timesteps"""
        self.aircraft = new_aircraft_config

    def change_control_sys(self, new_control_sys= None):
        """Changes the control system for future timesteps"""
        self.control_sys = new_control_sys

    def update_manual_control(self, stick_x, stick_y):
        """pass in a control vector for the simulation"""
        command = np.array([0.0, stick_x, stick_y, 0.0],'d')
        self.pilot_vec = command

    def pause_sim(self):
        """Pauses the sim, saving time at stop"""
        if not self.paused:
            self.paused = True
            self.time_at_last_pause = self.elapsed_time

    def unpause_sim(self):
        """Unpauses sim, starts counting time again"""
        if self.paused:
            self.paused = False
            self.start_time = time.time()

    def pause_or_unpause_sim(self):
        """Flips state of sim"""
        
        if self.paused:
            self.unpause_sim()
            print("unpause")
        else:
            self.pause_sim()
            print("pause")

    def run_sim(self):
        """runs the sim until t_span"""
        try:
            while self.time < self.t_span[1]:
                self.advance_timestep()
            self.time_at_last_pause = self.t_span[1]
        except KeyboardInterrupt:
            print("Simulation interuppted at t = ", self.time)
            exit(0)

    def pump_sim(self):
        """pumps the sim once, used for JIT compilation"""
        self.advance_timestep()

    def return_results(self):
        """logger"""
        return self.sim_log.return_data()

    def return_time_steps(self):
        """returns number of timesteps saved"""
        return self.sim_log.return_data_size()
