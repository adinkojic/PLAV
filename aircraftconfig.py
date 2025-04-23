"""Aircraft config file with helper functions
Abstracted away from any special state config"""
import math

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba import jit, types
import quaternion_math as quat


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

    ('C_XYlutX', float64[:]),
    ('C_XlutY', float64[:]),
    ('C_YlutY', float64[:]),

    ('has_gridfins', int64),
    ('top_force',  float64[:]),
    ('star_force', float64[:]),
    ('port_force', float64[:])

]

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

    #init_control_vector = np.zeros(4,'d')

    #C_XYlutX = np.array([0],'d')
    #C_XlutY  = np.array([0],'d')
    #C_YlutY  = np.array([0],'d')
    #inital trim controls
    if config_file['has_control']:
        init_control_vector =  np.array(config_file['init_control'],'d')

        #for glider with grid fins
        if config_file['hasgridfins']:
            C_XYlutX = np.array(config_file['C_XYlutX'],'d')
            C_XlutY  = np.array(config_file['C_XlutY'], 'd')
            C_YlutY  = np.array(config_file['C_YlutY'], 'd')

            print('wit gf')
            aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, C_D0, epsilon, C_m0, C_ma, C_mq,\
                 C_Yb, C_l, C_lp, C_lr, C_np, C_nr, C_mbb, C_Db, C_nb, init_control_vector, 1, C_XYlutX, C_XlutY, C_YlutY)
        else:
            print('no gf')
            aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, C_D0, epsilon, C_m0, C_ma, C_mq,\
                    C_Yb, C_l, C_lp, C_lr, C_np, C_nr, C_mbb, C_Db, C_nb, init_control_vector)
    else:
        print('no control')
        aircraft_model = AircraftConfig(mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, C_D0, epsilon, C_m0, C_ma, C_mq,\
                     C_Yb, C_l, C_lp, C_lr, C_np, C_nr, C_mbb, C_Db, C_nb)
        

    return aircraft_model

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

@jit(float64[:](float64, float64))
def get_wind_to_body_axis(alpha, beta):
    """Gets velocity to body axis, assumes x direction is datum"""
    beta_rot  = quat.from_angle_axis(-beta, np.array([0.0, 0.0, 1.0]))
    alpha_rot = quat.from_angle_axis(alpha, np.array([0.0, 1.0, 0.0]))
    result = quat.mulitply(beta_rot, alpha_rot)

    return result

@jit#(float64[:](float64[:], float64, float64))
def get_local_alpha_beta(velocity, gamma, theta):
    """Gets the local Alpha and Beta of a fin
    velocity is the local velocity [m/s]
    gamma is the rotation around +X, 0 is starboard [rad]
    theta is the angle of deflection of the surface [rad]"""

    x_chord_prime = np.array([np.cos(theta), 0.0, -np.sin(theta)], 'd')
    x_normal_prime = np.array([np.sin(theta), 0.0, np.cos(theta)], 'd')
    x_radial_prime = np.array([0.0, 1.0, 0.0], 'd')

    rotation_around_body = get_x_rotation_matrix(gamma)

    x_chord  = rotation_around_body @ x_chord_prime
    x_normal = rotation_around_body @ x_normal_prime
    x_radial = rotation_around_body @ x_radial_prime

    u = np.dot(velocity, x_chord)
    v = np.dot(velocity, x_radial)
    w = np.dot(velocity, x_normal)

    local_velocity = np.array([u, v, w], 'd')

    #airspeed, alpha, beta
    aab = velocity_to_alpha_beta(local_velocity)

    return np.array([aab[1], aab[2]], 'd')

@jit#(float64[:,:](float64))
def get_x_rotation_matrix(angle):
    """Gets a rotation matrix about X, useful for fins [rad]"""
    rotation_around_body = np.array([ [1., 0, 0], \
                         [0, math.cos(angle), -math.sin(angle)], \
                         [0, math.sin(angle), math.cos(angle)] ], 'd')
    return np.ascontiguousarray(rotation_around_body)

@jitclass(spec)
class AircraftConfig(object):
    """Aircraft jit'd object, responsible for storing all aircraft
    information and even giving forces"""

    def __init__(self, mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, C_D0, epsilon, C_m0, C_ma, C_mq,\
                C_Yb, C_l, C_lp, C_lr, C_np, C_nr, C_mbb, C_Db, C_nb, \
                init_control_vector = np.zeros(4), has_gridfins = 0, \
                C_XYlutX = np.array([0.0, 0.0]), C_XlutY =np.array([0.0, 0.0]), C_YlutY = np.array([0.0, 0.0])):
        self.mass = mass
        self.inertiamatrix = np.ascontiguousarray(inertia)
        self.cmac = cmac
        self.Sref = Sref
        self.bref = bref
        self.cp_wrt_cm = cp_wrt_cm

        self.trim_rdr   = init_control_vector[0]
        self.trim_ail   = init_control_vector[1]
        self.trim_el    = init_control_vector[2]
        self.trim_power = init_control_vector[3]

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

        self.C_XYlutX = C_XYlutX
        self.C_XlutY  = C_XlutY
        self.C_YlutY  = C_YlutY

        self.has_gridfins = has_gridfins
        self.top_force  = np.array([0.,0.,0.], 'd')
        self.star_force = np.array([0.,0.,0.], 'd')
        self.port_force = np.array([0.,0.,0.], 'd')

    def update_control(self, control_vector):
        """Give the simulation a new control vector"""
        self.rdr   = control_vector[0]
        self.ail   = control_vector[1]
        self.el    = control_vector[2]
        self.power = control_vector[3]

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
        body_forces_body = quat.rotateVectorQ(wind_to_body, body_forces_wind)

        aero_moments = np.array([body_rolling_moment, body_pitching_moment, body_yawing_moment])

        moments_with_torque = np.array([
            aero_moments[0] - self.cp_wrt_cm[2]*body_forces_body[1] + self.cp_wrt_cm[1]*body_forces_body[2],
            aero_moments[1] + self.cp_wrt_cm[2]*body_forces_body[0] - self.cp_wrt_cm[0]*body_forces_body[2],
            aero_moments[2] - self.cp_wrt_cm[1]*body_forces_body[0] + self.cp_wrt_cm[0]*body_forces_body[1],
        ], 'd')

        if self.has_gridfins == 1:
            gridfin_forces, gridfin_moments = self.calculate_grid_fin_forces()
            body_forces_body = body_forces_body + gridfin_forces
            moments_with_torque = moments_with_torque + gridfin_moments

        return body_forces_body, moments_with_torque

    def calculate_grid_fin_forces(self):
        """Calculates the forces of each grid fin"""
        qbar = self.get_qbar()
        S = self.Sref

        top_fin_gamma  = -math.pi/2.0
        star_fin_gamma =  math.pi/6.0
        port_fin_gamma =  5.0*math.pi/6.0

        yaw_adjustment_factor = 0.5

        ail_command = self.ail * 0.5 + self.trim_ail
        el_command  = self.el  * 0.5 + self.trim_el
        rdr_command = self.rdr * 0.5 + self.trim_rdr

        top_fin_theta  = -ail_command + rdr_command
        star_fin_theta = -ail_command + rdr_command*yaw_adjustment_factor + el_command
        port_fin_theta = -ail_command + rdr_command*yaw_adjustment_factor - el_command

        aab_top  = get_local_alpha_beta(self.velocity, top_fin_gamma,  top_fin_theta )
        aab_star = get_local_alpha_beta(self.velocity, star_fin_gamma, star_fin_theta)
        aab_port = get_local_alpha_beta(self.velocity, port_fin_gamma, port_fin_theta)


        top_drag_angle  = np.sqrt(aab_top[0]**2  + aab_top[1]**2 )
        star_drag_angle = np.sqrt(aab_star[0]**2 + aab_star[1]**2)
        port_drag_angle = np.sqrt(aab_port[0]**2 + aab_port[1]**2)

        top_drag  = np.interp(top_drag_angle,  self.C_XYlutX, self.C_XlutY)
        star_drag = np.interp(star_drag_angle, self.C_XYlutX, self.C_XlutY)
        port_drag = np.interp(port_drag_angle, self.C_XYlutX, self.C_XlutY)


        top_normal  = np.interp(aab_top[0],  self.C_XYlutX, self.C_YlutY)
        top_radial  = np.interp(aab_top[1],  self.C_XYlutX, self.C_YlutY) * 0.5
        star_normal = np.interp(aab_star[0], self.C_XYlutX, self.C_YlutY)
        star_radial = np.interp(aab_star[1], self.C_XYlutX, self.C_YlutY) * 0.5
        port_normal = np.interp(aab_port[0], self.C_XYlutX, self.C_YlutY)
        port_radial = np.interp(aab_port[1], self.C_XYlutX, self.C_YlutY) * 0.5

        top_rot  = get_x_rotation_matrix(top_fin_gamma )
        star_rot = get_x_rotation_matrix(star_fin_gamma)
        port_rot = get_x_rotation_matrix(port_fin_gamma)

        top_forces_ring  = np.array([-top_drag,  -top_radial, -top_normal],'d')
        star_forces_ring = np.array([-star_drag, -star_radial,-star_normal],'d')
        port_forces_ring = np.array([-port_drag, -port_radial,-port_normal],'d')

        #from coeff to real force
        top_force  = qbar * S * top_rot  @ top_forces_ring
        star_force = qbar * S * star_rot @ star_forces_ring
        port_force = qbar * S * port_rot @ port_forces_ring

        grid_fin_arm = np.array([-0.5, 0.0889, 0.0], 'd')

        top_arm  = top_rot  @ grid_fin_arm
        star_arm = star_rot @ grid_fin_arm
        port_arm = port_rot @ grid_fin_arm

        top_moment  = np.cross(top_arm,  top_force)
        star_moment = np.cross(star_arm, star_force)
        port_moment = np.cross(port_arm, port_force)

        total_force = top_force + star_force + port_force

        total_moment= top_moment+ star_moment+ port_moment

        return total_force, total_moment

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
