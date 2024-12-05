import math
import numpy as np
from numba import jit
import quaternion_math as quat


'''BRGR Glide Vehicle Aero Sim
TODO: simulation for high altitude'''

#gives in body frame
def from_alpha_beta(airspeed, alpha, beta):
    x = airspeed * math.cos(alpha * math.pi/180) * math.cos(beta * math.pi/180)
    y = airspeed * math.sin(beta  * math.pi/180)
    z = airspeed * math.sin(-alpha * math.pi/180) * math.cos(beta * math.pi/180)

    return np.array([x,y,z])

@jit #velocity, sans wind
def velocity_to_alpha_beta(velocity_body):
    airspeed = math.sqrt(velocity_body[0]**2 + velocity_body[1]**2 + velocity_body[2]**2)
    temp = math.sqrt(velocity_body[0]**2 + velocity_body[2]**2)
    beta = math.atan2(velocity_body[1], temp)
    alpha = -math.atan2(velocity_body[2], velocity_body[0])

    return airspeed, alpha, beta

#TODO: implement atmosphere model
@jit
def get_atmosphere(altitude):
    density = 1.225
    viscosity = 0.00001789
    return density, viscosity

@jit
def get_Re(airspeed, density, viscosity):
    length = 0.2 #characteristic length
    Re = airspeed * length * density/viscosity
    return Re

#TODO: implement based on better data and with Re
@jit
def get_coeff(alpha, beta, Re, p, q ,r):
    aa = np.array([alpha]).clip(-15*math.pi/180, 15*math.pi/180)
    alpha = aa[0]

    C_L0 = 0.34
    C_La = 4.
    C_D0 = 0.022
    epsilon = 0.069
    C_M0 = 0.045/2
    C_Ma = .12/2
    C_Mq = -0.01 #not sure about this value

    C_L = C_L0 + C_La * alpha
    C_D = C_D0 + epsilon * C_L**2
    C_M = C_M0 + C_Ma * alpha #+ C_Mq * q

    C_Y = 0 #yaw
    C_R = 0 #roll
    C_Z = 0 #side force

    return C_L,C_D,C_M, C_Z, C_Y, C_R

@jit
def get_wind_to_stability_axis(alpha, beta):
    beta_rot  = quat.from_angle_axis(beta, np.array([0, 0, 1]))
    alpha_rot = quat.from_angle_axis(beta, np.array([0, -1, 0]))

    return quat.mulitply(alpha_rot, beta_rot)

#TODO: test and implement grid fins
@jit
def get_aero_forces(state):
    Aref = 0.2 #m**2
    mac = 0.1778 #m

    airspeed, alpha, beta = velocity_to_alpha_beta(np.array([state[3],state[4],state[5]]))

    altitude = state[2]

    density, viscosity = get_atmosphere(altitude)
    Re = get_Re(airspeed, density, viscosity)

    p, q, r = state[10], state[11], state[12]

    C_L, C_D, C_M, C_Z, C_Y, C_R = get_coeff(alpha, beta, Re, p, q ,r)

    qbar = 0.5 * density *airspeed**2

    body_lift = C_L * qbar * Aref
    body_drag = C_D * qbar * Aref
    body_side = C_Z * qbar * Aref
    body_pitching_moment = C_M * qbar * Aref * mac
    body_yawing_moment   = C_Y * qbar * Aref #Coeff definition might be wrong
    body_rolling_moment  = C_R * qbar * Aref

    wind_to_stab = get_wind_to_stability_axis(alpha,beta)

    body_forces_wind = np.array([-body_drag, body_side, body_lift])
    body_forces_stab = quat.rotateVectorQ(wind_to_stab, body_forces_wind)

    moments = np.array([body_rolling_moment, -body_pitching_moment, body_yawing_moment])

    return body_forces_stab, moments
    




