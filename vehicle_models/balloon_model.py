"""Basic balloon model"""
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
    ('gravity', float64[:]),

    
    ('C_D0', float64),
    ('epsilon', float64),
    

    ('C_mq', float64),
    ('C_lp', float64),
    ('C_nr', float64),

    #balloon specific
    ('gas_cf', float64),
    ('burst_dia_ft', float64),
    ('burst_flag', float64),
    ('balloon_volume', float64)
]

@jit(float64(float64))
def get_dynamic_viscosity(temperature):
    """Equation 51 of USSA1976"""
    beta = 1.458e-6
    sutherlands = 110.4

    mu = (beta * temperature**1.5)/(temperature+sutherlands)

    return mu

@jit(float64[:](float64[:]))
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

@jit(float64[:](float64,float64,float64))
def alpha_beta_to_velocity(airspeed, alpha, beta):
    """Turns airspeed, alpha, and beta to UVW values"""
    u = airspeed * math.cos(beta) * math.cos(alpha)
    v = airspeed * math.sin(beta)
    w = airspeed * math.cos(beta) * math.sin(alpha)

    return np.array([u, v, w], 'd')

@jit(float64[:](float64, float64))
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

    def __init__(self, mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_D0, C_mq, C_lp, C_nr, gas_cf, burst_dia_ft):
        self.mass = mass
        self.inertiamatrix = np.ascontiguousarray(inertia)
        self.cmac = cmac
        self.Sref = Sref
        self.bref = bref
        self.cp_wrt_cm = cp_wrt_cm

        
        self.C_D0 = C_D0
        self.C_lp = C_lp
        self.C_mq = C_mq
        self.C_nr = C_nr

        self.altitude = 0.0
        self.velocity = np.zeros(3, 'd')
        self.omega = np.zeros(3,'d')
        self.airspeed = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.reynolds = 0.0
        self.density = 0.0
        self.temperature = 0.0
        self.mach = 0.0
        self.gravity = np.zeros(3, 'd')

        self.gas_cf = gas_cf
        self.burst_dia_ft = burst_dia_ft
        self.burst_flag = 0
        self.balloon_volume = gas_cf / 35.315

    def update_control(self, rudder, aileron, elevator, throttle):
        """pass"""
        

    def update_trim(self, rudder, aileron, elevator, throttle):
        """pass"""

    def update_conditions(self, altitude, velocity, omega, density, temperature, speed_of_sound, gravity):
        """Update altitude and velocity it thinks it's at
        Call this before every get_forces()"""
        self.altitude = altitude
        self.velocity = velocity
        self.omega = omega
        self.gravity = gravity

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
        if self.airspeed**2 < 0.1: #avoids div/0
            p_hat = 0
            q_hat = 0
            r_hat = 0
        else:
            p_hat = self.bref * p/2/self.airspeed
            q_hat = self.cmac * q/2/self.airspeed
            r_hat = self.bref * r/2/self.airspeed

        C_L = 0.0
        C_D = self.C_D0
        C_m = self.C_mq * q_hat
        # + self.C_ma * self.alpha this is covered by crossing forces with x_cp

        C_Y = 0.0
        C_l = self.C_lp * p_hat
        C_n = self.C_nr * r_hat

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

        body_forces_body = body_forces_body + self.get_buoyancy_force()

        moments_with_torque = np.array([
            aero_moments[0] - self.cp_wrt_cm[2]*body_forces_body[1] + self.cp_wrt_cm[1]*body_forces_body[2],
            aero_moments[1] + self.cp_wrt_cm[2]*body_forces_body[0] - self.cp_wrt_cm[0]*body_forces_body[2],
            aero_moments[2] - self.cp_wrt_cm[1]*body_forces_body[0] + self.cp_wrt_cm[0]*body_forces_body[1],
        ], 'd')

        return body_forces_body, moments_with_torque

    def get_buoyancy_force(self):
        """Solves for buoyancy forces"""

        pressure_sea = 101_325.0 #Pa
        temperature_sea = 288.15 #K
        density_sea = (287.05 * temperature_sea) /pressure_sea

        volume_sea = self.gas_cf / 35.315 #cubic feet to m^3
        burst_volume = (4/3) * np.pi * (self.burst_dia_ft / 3.281 / 2)**3

        amb_pressure = 287.05 * self.temperature * self.density

        self.balloon_volume = pressure_sea * volume_sea * temperature_sea /self.temperature /amb_pressure

        self.Sref = (self.balloon_volume /(4/3) / np.pi)**(2/3) * np.pi #approx cross sectional area from volume

        if self.balloon_volume > burst_volume :
            self.burst_flag = 1

        if self.burst_flag == 1:
            self.Sref = 10.0
            return np.zeros(3,'d')

        bouancy_force = self.density * self.balloon_volume * -self.gravity

        return bouancy_force
    
    def get_area(self):
        """Returns Sref"""
        return self.Sref

    def get_burst_flag(self):
        """Returns burst flag"""
        return self.burst_flag
    
    def get_balloon_diameter(self):
        """returns current diameter [ft]"""
        return (self.balloon_volume * 35.315 /(4/3) / np.pi)**(1/3) * 2 #diameter from volume

    def get_control_deflection(self):
        """Returns the current control state"""
        return np.zeros(4, 'd')

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

    C_D0 = config_file['C_D0']
    C_mq = config_file['C_mq']
    C_lp = config_file['C_lp']
    C_nr = config_file['C_nr']


    cp_wrt_cm = np.array( config_file['xcp_wrt_cm'])

    gas_cf = config_file['gas_cf']
    burst_dia_ft = config_file['burst_dia_ft']

    aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_D0, C_mq, C_lp, \
                                    C_nr, gas_cf, burst_dia_ft)

    return aircraft_model

