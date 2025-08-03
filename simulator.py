"""
Simulation object lives here, along with the x_dot function

Gravity lives here as well (J2)

"""

import math
import numpy as np
#from scipy.integrate import solve_ivp
from numba import jit, float64

import quaternion_math as quat
from step_logging import SimDataLogger
from runge_kutta4 import basic_rk4

@jit(float64(float64,float64), cache=True)
def get_gravity(phi, h):
    """gets gravity accel from lat and altitude
    phi: latitude [rad]
    h: altitude [m]
    returns: gravity acceleration [m/s^2]"""
    gravity = 9.780327*(1 + 5.3024e-3*np.sin(phi)**2 - 5.8e-6*np.sin(2*phi)**2) \
            - (3.0877e-6 - 4.4e-9*np.sin(phi)**2)*h + 7.2e-14*h**2
    return gravity

@jit
def x_dot(t, y, aircraft_config, atmosphere, log = None):
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

    atmosphere.update_conditions(altitude)

    air_density = atmosphere.get_density()
    air_temperature = atmosphere.get_temperature()
    static_pressure = atmosphere.get_pressure()
    speed_of_sound = atmosphere.get_speed_of_sound()

    #adds wind
    v_airspeed = quat.rotateVectorQ(q, np.array([vn, ve, vd], 'd') + atmosphere.get_wind_ned())
    #solving for acceleration, which is velocity_dot
    aircraft_config.update_conditions(altitude,  v_airspeed, omega, air_density, air_temperature, speed_of_sound)


    aero_forces_body, aero_moments = aircraft_config.get_forces()
    aircraft_thrust = aircraft_config.calculate_thrust()

    body_forces_body = np.array([
        aero_forces_body[0] + aircraft_thrust,
        aero_forces_body[1],
        aero_forces_body[2],
        ], 'd')

    accel_body = body_forces_body/mass

    accel_ned = quat.rotateFrameQ(q, accel_body)


    accel_north = accel_ned[0]
    accel_east  = accel_ned[1]
    accel_down  = accel_ned[2]

    omega = omega - quat.rotateFrameQ(q, omega_NI)

    #integrate state
    q1dot = 0.5*(-omega[0]*q[1] -omega[1]*q[2] -omega[2]*q[3])
    q2dot = 0.5*( omega[0]*q[0] +omega[2]*q[2] -omega[1]*q[3])
    q3dot = 0.5*( omega[1]*q[0] -omega[2]*q[1] +omega[0]*q[3])
    q4dot = 0.5*( omega[2]*q[0] +omega[1]*q[1] -omega[0]*q[2])

    #(11.27) in Engineeering Dyanmics (Kasdin and Paley)
    omega_dot = np.linalg.solve(inertia_tensor, aero_moments - np.cross(np.eye(3), omega) @ inertia_tensor @ omega)

    lat_dot = vn/(R_phi+altitude)
    long_dot = ve/((R_lamb+altitude)*math.cos(lat))
    altitude_dot = -vd

    #from book Optimal Estimation of Dynamic Systems
    vn_dot = accel_north-(long_dot + 2*omega_e)*ve*np.sin(lat) + vn*vd/(R_phi+altitude)
    ve_dot = accel_east -(long_dot + 2*omega_e)*vn*np.sin(lat) + ve*vd/(R_phi+altitude) + 2*omega_e*vd*np.cos(lat)
    vd_dot = accel_down + gravity-ve**2/(R_lamb+altitude)-vn**2/(R_phi+altitude) - 2*omega_e*ve*np.cos(lat)



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
                    alpha, beta, reynolds, aircraft_thrust, control_deflection)

    return x_dot