"""Aircraft config file with helper functions
Abstracted away from any special state config"""
import math

import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba import jit
from plav.quaternion_math import from_angle_axis, mulitply, rotateVectorQ

spec = [
    #geometrics
    ('mass', float64),
    ('cmac', float64),
    ('Sref', float64),
    ('bref', float64),
    ('inertiamatrix', float64[:,:]),
    ('cp_wrt_cm', float64[:]),

    ('rdr', float64),
    ('ail', float64),
    ('el', float64),
    ('power', float64),

    ('trim_rdr', float64),
    ('trim_ail', float64),
    ('trim_el', float64),
    ('trim_power', float64),

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
    ('mach', float64),

    #areodynamics
    ('C_L0', float64),
    ('C_La', float64),
    ('C_D0', float64),
    ('epsilon', float64),
    ('C_m0', float64),
    ('C_ma', float64),
    ('C_mq', float64),
    ('C_mbb', float64),
    ('C_Yb', float64),
    ('C_R', float64),
    ('C_Z', float64),
    ('C_Db', float64),
    ('C_nb', float64),

    ('C_Yb', float64),
    ('C_l', float64),
    ('C_lp', float64),
    ('C_lr', float64),
    ('C_np', float64),
    ('C_nr', float64),

]

@jit(float64(float64),cache=True)
def get_dynamic_viscosity(temperature):
    """Equation 51 of USSA1976"""
    beta = 1.458e-6
    sutherlands = 110.4

    mu = (beta * temperature**1.5)/(temperature+sutherlands)

    return mu

@jit(float64[:](float64[:]),cache=True)
def velocity_to_alpha_beta(velocity_body):
    """Gets velocity to alpha beta, assumes x direction is datum
    Reference: Fundamentals of Helicopter Dyanmics 10.42, 10.43"""
    airspeed = math.sqrt(velocity_body[0]**2 + velocity_body[1]**2 + velocity_body[2]**2)

    if abs(airspeed) > 0.01:
        beta = math.asin(velocity_body[1]/airspeed)
    else:
        beta = 0.0
    alpha = math.atan2(velocity_body[2], velocity_body[0])

    return np.array([airspeed, alpha, beta], 'd')

@jit(float64[:](float64,float64,float64),cache=True)
def alpha_beta_to_velocity(airspeed, alpha, beta):
    """Turns airspeed, alpha, and beta to UVW values"""
    u = airspeed * math.cos(beta) * math.cos(alpha)
    v = airspeed * math.sin(beta)
    w = airspeed * math.cos(beta) * math.sin(alpha)

    return np.array([u, v, w], 'd')

@jit(float64[:](float64, float64),cache=True)
def get_wind_to_body_axis(alpha, beta):
    """Gets velocity to body axis, assumes x direction is datum"""
    beta_rot  = from_angle_axis(-beta, np.array([0.0, 0.0, 1.0]))
    alpha_rot = from_angle_axis(alpha, np.array([0.0, 1.0, 0.0]))
    result = mulitply(beta_rot, alpha_rot)

    return result

@jitclass(spec)
class AircraftConfig(object):
    """Aircraft jit'd object, responsible for storing all aircraft
    information and even givings forces"""

    def __init__(self, mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, C_D0, epsilon, \
                C_m0, C_ma, C_mq, C_Yb, C_l, C_lp, C_lr, C_np, C_nr, C_mbb, C_Db, C_nb, \
                trim_rudder = 0, trim_aileron = 0, trim_elevator = 0, trim_throttle = 0):
        self.mass = mass
        self.inertiamatrix = np.ascontiguousarray(inertia)
        self.cmac = cmac
        self.Sref = Sref
        self.bref = bref
        self.cp_wrt_cm = cp_wrt_cm

        self.trim_rdr   = trim_rudder
        self.trim_ail   = trim_aileron
        self.trim_el    = trim_elevator
        self.trim_power = trim_throttle

        self.rdr   = 0.0
        self.ail   = 0.0
        self.el    = 0.0
        self.power = 0.0

        self.C_L0 = C_L0
        self.C_La = C_La
        self.C_D0 = C_D0
        self.epsilon = epsilon
        self.C_m0 = C_m0
        self.C_ma = C_ma
        self.C_mq = C_mq
        self.C_mbb= C_mbb
        self.C_Db = C_Db

        self.C_Yb = C_Yb
        self.C_l  = C_l
        self.C_lp = C_lp
        self.C_lr = C_lr
        self.C_np = C_np
        self.C_nr = C_nr
        self.C_nb = C_nb


        self.altitude = 0.0
        self.velocity = np.zeros(3, 'd')
        self.omega = np.zeros(3)
        self.airspeed = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.reynolds = 0.0
        self.density = 0.0
        self.temperature = 0.0
        self.mach = 0.0

    def update_control(self, rudder, aileron, elevator, throttle):
        """Give the simulation a new control vector"""
        self.rdr   = rudder
        self.ail   = aileron
        self.el    = elevator
        self.power = throttle

    def update_trim(self, rudder, aileron, elevator, throttle):
        """Give the simulation a new trim vector"""
        self.trim_rdr   = rudder
        self.trim_ail   = aileron
        self.trim_el    = elevator
        self.trim_power = throttle

    def update_conditions(self, altitude, velocity, omega, density, temperature, speed_of_sound):
        """Update altitude and velocity it thinks it's at
        Call this before every get_forces()"""
        self.altitude = altitude
        self.velocity = velocity
        self.omega = omega

        self.density = density
        self.temperature = temperature

        aab = velocity_to_alpha_beta(velocity)
        self.airspeed = aab[0]
        self.alpha = aab[1]
        self.beta = aab[2]

        dynamic_viscosity = get_dynamic_viscosity(temperature)
        self.reynolds = self.get_Re(density, dynamic_viscosity)

        self.mach = self.airspeed/speed_of_sound

    def get_coeff(self):
        """Gets aircraft aero coeff from given conditions"""

        p, q, r = self.omega[0], self.omega[1], self.omega[2]

        #non-dimensional airspeed
        if abs(self.airspeed) < 0.1: #avoids div/0
            p_hat = 0
            q_hat = 0
            r_hat = 0
        else:
            p_hat = self.bref * p/2/self.airspeed
            q_hat = self.cmac * q/2/self.airspeed
            r_hat = self.bref * r/2/self.airspeed

        C_L = self.C_L0 + self.C_La * self.alpha #this needs to be limited but it isnt working right
        C_D = self.C_D0 + self.epsilon * C_L**2 + self.C_Db * abs(self.beta)
        C_m = self.C_m0 + self.C_mq * q_hat + self.C_mbb * self.beta ** 2
        # + self.C_ma * self.alpha this is covered by crossing forces with x_cp

        C_Y = self.C_Yb * self.beta #side force
        C_l = self.C_l + self.C_lr * r_hat + self.C_lp * p_hat #roll
        C_n = self.C_np * p_hat + self.C_nr * r_hat # -self.C_nb * self.beta#yaw force

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

        wind_to_body = get_wind_to_body_axis(self.alpha, self.beta)

        body_forces_wind = np.array([-body_drag, body_side, -body_lift])
        body_forces_body = rotateVectorQ(wind_to_body, body_forces_wind)

        aero_moments = np.array([body_rolling_moment, body_pitching_moment, body_yawing_moment])

        moments_with_torque = np.array([
            aero_moments[0] - self.cp_wrt_cm[2]*body_forces_body[1] + self.cp_wrt_cm[1]*body_forces_body[2],
            aero_moments[1] + self.cp_wrt_cm[2]*body_forces_body[0] - self.cp_wrt_cm[0]*body_forces_body[2],
            aero_moments[2] - self.cp_wrt_cm[1]*body_forces_body[0] + self.cp_wrt_cm[0]*body_forces_body[1],
        ], 'd')

        return body_forces_body, moments_with_torque

    def get_control_deflection(self):
        """Returns the current control state"""
        return np.array([(self.rdr + self.trim_rdr), \
                        (self.ail + self.trim_ail), \
                        (self.el + self.trim_el), \
                        (self.power + self.trim_power)], 'd')

    def calculate_thrust(self):
        """dummy for now, returns 0.0"""
        return 0.0

    def get_xcp(self):
        """returns x_cp with respect to CM"""
        return self.cp_wrt_cm

    def get_inertia_matrix(self):
        """Returns inertia matrix as 2d np array"""
        return np.ascontiguousarray(self.inertiamatrix)

    def get_mass(self):
        """Returns mass in kg"""
        return self.mass

    def get_Re(self, density, viscosity):
        """Gets reynolds number from given conditions"""
        return self.airspeed * self.cmac * density/viscosity

    def get_alpha(self):
        """Returns alpha in rad"""
        return self.alpha

    def get_beta(self):
        """Returns beta in rad"""
        return self.beta

    def get_mach(self):
        """Returns mach [nd]"""
        return self.mach

    def get_qbar(self):
        """Returns dyanmic pressure [Pa]"""
        return 0.5 * self.density *self.airspeed**2

    def get_airspeed(self):
        """Returns airspeed [m/s]"""
        return self.airspeed

    def get_reynolds(self):
        """Returns Reynolds Number"""
        return self.reynolds

def init_aircraft(config_file) -> AircraftConfig:
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
    C_Db = config_file['C_Db']

    C_Yb  = config_file['C_Yb']
    C_l  = config_file['C_l']
    C_lp = config_file['C_lp']
    C_lr = config_file['C_lr']
    C_np = config_file['C_np']
    C_nr = config_file['C_nr']
    C_nb = config_file['C_nb']

    C_mbb = config_file['C_mbb']


    cp_wrt_cm = np.array( config_file['xcp_wrt_cm'])


    if config_file['has_control']:
        init_control_vector =  np.array(config_file['init_control'],'d')
        aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, \
                                        C_D0, epsilon, C_m0, C_ma, C_mq,C_Yb, C_l, C_lp, C_lr, \
                                        C_np, C_nr, C_mbb, C_Db, C_nb, init_control_vector)
    else:
        aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, \
                                        C_D0, epsilon, C_m0, C_ma, C_mq, C_Yb, C_l, C_lp, C_lr, \
                                        C_np, C_nr, C_mbb, C_Db, C_nb)

    return aircraft_model

def init_dummy_aircraft() -> AircraftConfig:
    """Returns a dummy aircraft config for initialization"""

    return AircraftConfig(mass = 1.0, inertia = np.eye(3,'d'), cmac = 1.0, Sref = 1.0, bref = 1.0,\
                        cp_wrt_cm = np.array([1.0, 0.0, 0.0], 'd'), C_L0 = 0.0, C_La = 0.0, \
                        C_D0 = 0.0, epsilon = 0.0, C_m0 = 0.0, C_ma = 0.0, C_mq = 0.0, C_Yb = 0.0,\
                        C_l = 0.0, C_lp = 0.0, C_lr = 0.0, C_np = 0.0, C_nr = 0.0, C_mbb = 0.0, \
                        C_Db = 0.0, C_nb = 0.0)
