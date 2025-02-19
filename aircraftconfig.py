"""Aircraft config file with helper functions
Abstracted away from any special state config"""
import math

import numpy as np
from numba import float64, bool    # import the types
from numba.experimental import jitclass
from numba import jit

import quaternion_math as quat


spec = [
    #geometrics
    ('mass', float64),
    ('cmac', float64),
    ('Sref', float64),
    ('bref', float64),

    #enviromentals
    ('altitude', float64),
    ('velocity', float64[:]),
    ('airspeed', float64),
    ('alpha', float64),
    ('beta', float64),
    ('reynolds', float64),
    ('omega', float64[:]),
    ('density', float64),
    ('temperature', float64),

    #areodynamics
    ('C_L0', float64),
    ('C_La', float64),
    ('C_D0', float64),
    ('epsilon', float64),
    ('C_m0', float64),
    ('C_ma', float64),
    ('C_mq', float64),
    ('C_Y', float64),
    ('C_R', float64),
    ('C_Z', float64),
    ('inertiamatrix', float64[:,:]),

    ('C_Y', float64),
    ('C_l', float64),
    ('C_lp', float64),
    ('C_lr', float64),
    ('C_np', float64),
    ('C_nr', float64),

]

@jit
def get_dynamic_viscosity(temperature):
    """Equation 51 of USSA1976"""
    beta = 1.458e-6
    sutherlands = 110.4

    mu = (beta * temperature**1.5)/(temperature+sutherlands)

    return mu

@jit
def velocity_to_alpha_beta(velocity_body):
    """Gets velocity to alpha beta, assumes x direction is datum"""
    airspeed = math.sqrt(velocity_body[0]**2 + velocity_body[1]**2 + velocity_body[2]**2)
    temp = math.sqrt(velocity_body[0]**2 + velocity_body[2]**2)
    beta = math.atan2(velocity_body[1], temp)
    alpha = -math.atan2(velocity_body[2], velocity_body[0])

    return airspeed, alpha, beta

@jit
def get_wind_to_stability_axis(alpha, beta):
    """Gets velocity to stability axis, assumes x direction is datum"""
    beta_rot  = quat.from_angle_axis(beta, np.array([0, 0, 1]))
    alpha_rot = quat.from_angle_axis(alpha, np.array([0, -1, 0]))

    return quat.mulitply(alpha_rot, beta_rot)

@jitclass(spec)
class AircraftConfig(object):
    """Aircraft jit'd object, responsible for storing all aircraft
    information and even giving forces"""

    def __init__(self,mass, inertia, cmac, Sref, bref, C_L0, C_La, C_D0, epsilon, C_m0, C_ma, C_mq,\
                 C_Y, C_l, C_lp, C_lr, C_np, C_nr ):
        self.mass = mass
        self.inertiamatrix = inertia
        self.cmac = cmac
        self.Sref = Sref
        self.bref = bref

        self.C_L0 = C_L0
        self.C_La = C_La
        self.C_D0 = C_D0
        self.epsilon = epsilon
        self.C_m0 = C_m0
        self.C_ma = C_ma
        self.C_mq = C_mq

        self.C_Y  = C_Y
        self.C_l  = C_l
        self.C_lp  = C_lp
        self.C_lr  = C_lr
        self.C_np  = C_np
        self.C_nr  = C_nr



        self.altitude = 0.0
        self.velocity = np.zeros(3)
        self.omega = np.zeros(3)
        self.airspeed = 0
        self.alpha = 0
        self.beta = 0
        self.reynolds = 0
        self.density = 0
        self.temperature = 0

    def update_conditions(self, altitude, velocity, omega, density, temperature):
        """Update altitude and velocity it thinks it's at
        Call this before every get_forces()"""
        self.altitude = altitude
        self.velocity = velocity
        self.omega = omega

        self.density = density
        self.temperature = temperature


        self.airspeed, self.alpha, self.beta = velocity_to_alpha_beta(velocity)

        dynamic_viscosity = get_dynamic_viscosity(temperature)
        self.reynolds = self.get_Re(density, dynamic_viscosity)


    def get_inertia_matrix(self):
        """Returns inertia matrix as 2d np array"""
        return self.inertiamatrix

    def get_mass(self):
        """Returns mass in kg"""
        return self.mass
    

    def get_Re(self, density, viscosity):
        """Gets reynolds number from given conditions"""
        return self.airspeed * self.cmac * density/viscosity

    def get_coeff(self):
        """Gets aircraft aero coeff from given conditions"""
        aa = np.array([self.alpha]).clip(-15*math.pi/180, 15*math.pi/180)
        alpha = aa[0]

        p, q, r = self.omega[0], self.omega[1], self.omega[2]

        C_L = self.C_L0 + self.C_La * alpha
        C_D = self.C_D0 + self.epsilon * C_L**2
        C_m = self.C_m0 + self.C_ma * alpha + self.C_mq * q

        C_Y = self.C_Y #side force
        C_l = self.C_l + self.C_lr * r + self.C_lp * p #roll TODO:beta dependence
        C_n = self.C_np * p + self.C_nr * r #yaw force, TODO: beta dependence

        return C_L,C_D,C_m, C_Y, C_l, C_n

    def get_forces(self):
        """Gets forces on aircraft from state and known derivatives"""


        C_L,C_D,C_m, C_Y, C_l, C_n = self.get_coeff()

        qbar = 0.5 * self.density *self.airspeed**2

        body_lift = C_L * qbar * self.Sref
        body_drag = C_D * qbar * self.Sref
        body_side = C_Y * qbar * self.Sref
        body_pitching_moment = C_m * qbar * self.Sref * self.cmac
        body_yawing_moment   = C_n * qbar * self.Sref * self.bref
        body_rolling_moment  = C_l * qbar * self.Sref * self.bref

        wind_to_stab = get_wind_to_stability_axis(self.alpha, self.beta)

        body_forces_wind = np.array([-body_drag, body_side, body_lift])
        body_forces_stab = quat.rotateVectorQ(wind_to_stab, body_forces_wind)

        moments = np.array([body_rolling_moment, body_pitching_moment, body_yawing_moment])

        return body_forces_stab, moments
