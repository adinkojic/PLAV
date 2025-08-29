"""Aircraft config file with helper functions
Abstracted away from any special state config"""
import math

import numpy as np
from numba import float64, int64, bool
from numba.experimental import jitclass
from numba import jit
import plav.quaternion_math as quat

from plav.vehicle_models.generic_aircraft_config import \
    get_dynamic_viscosity,get_wind_to_body_axis,velocity_to_alpha_beta

from plav.plav import load_scenario

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
    ('gravity', float64[:]),

    #areodynamics
    ('C_L0', float64),
    ('C_La', float64),
    ('C_Lmax', float64),
    ('C_Lmin', float64),
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
    ('port_force', float64[:]),

    ('plav_mixing', int64),
    ('on_balloon', int64),
    ('gas_cf', float64),
    ('burst_dia_ft', float64),
    ('burst_flag', float64),
    ('balloon_volume', float64),
    ('brgr_Sref', float64),
    ('brgr_mass', float64),

    ('prev_command', float64[:]),
    ('prev_position', float64[:]),

]

@jit
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

@jit
def get_x_rotation_matrix(angle):
    """Gets a rotation matrix about X, useful for fins [rad]"""
    rotation_around_body = np.array([ [1., 0.0, 0.0], \
                         [0.0, math.cos(angle), -math.sin(angle)], \
                         [0.0, math.sin(angle), math.cos(angle)] ], 'd')

    return rotation_around_body

@jitclass(spec)
class BRGRConfig(object):
    """Aircraft jit'd object, responsible for storing all aircraft
    information and even giving forces"""

    def __init__(self, mass, inertia, cmac, Sref, bref, cp_wrt_cm, C_L0, C_La, C_Lmax, C_Lmin, C_D0, epsilon, C_m0, C_ma,\
                    C_mq, C_Yb, C_l, C_lp, C_lr, C_np, C_nr, C_mbb, C_Db, C_nb, \
                    trim_rudder, trim_aileron, trim_elevator, trim_throttle, has_gridfins = 0, \
                    C_XYlutX = np.array([0.0, 0.0]), C_XlutY =np.array([0.0, 0.0]), \
                    C_YlutY = np.array([0.0, 0.0]), plav_mixing = 1, on_balloon = 0, \
                    gas_cf = 200, burst_dia_ft = 47.2):

        self.mass = mass
        self.inertiamatrix = np.ascontiguousarray(inertia)
        self.cmac = cmac
        self.Sref = Sref
        self.brgr_Sref = Sref
        self.brgr_mass = mass
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

        self.prev_command = np.zeros(3, 'd')
        self.prev_position = np.zeros(3, 'd')

        self.C_L0 = C_L0
        self.C_La = C_La
        self.C_Lmax = C_Lmax
        self.C_Lmin = C_Lmin
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
        self.omega = np.zeros(3, 'd')
        self.airspeed = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.reynolds = 0.0
        self.density = 0.0
        self.temperature = 0.0
        self.mach = 0.0
        self.gravity = np.zeros(3,'d')

        self.C_XYlutX = C_XYlutX
        self.C_XlutY  = C_XlutY
        self.C_YlutY  = C_YlutY

        self.has_gridfins = has_gridfins
        self.top_force  = np.array([0.,0.,0.], 'd')
        self.star_force = np.array([0.,0.,0.], 'd')
        self.port_force = np.array([0.,0.,0.], 'd')

        self.plav_mixing = plav_mixing
        self.on_balloon = on_balloon

        self.gas_cf = gas_cf
        self.burst_dia_ft = burst_dia_ft
        self.burst_flag = 0
        self.balloon_volume = gas_cf / 35.315
        

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
        if abs(self.airspeed) < 0.1: #avoids div/0
            p_hat = 0
            q_hat = 0
            r_hat = 0
        else:
            p_hat = self.bref * p/2/self.airspeed
            q_hat = self.cmac * q/2/self.airspeed
            r_hat = self.bref * r/2/self.airspeed

        C_L = self.C_L0 + self.C_La * self.alpha

        if C_L > self.C_Lmax:
            C_L = self.C_Lmax
        if C_L < self.C_Lmin:
            C_L = self.C_Lmin

        C_D = self.C_D0 + self.epsilon * C_L**2 + self.C_Db * abs(self.beta)
        C_m = self.C_m0 + self.C_mq * q_hat #+ self.C_mbb * self.beta ** 2
        # + self.C_ma * self.alpha this is covered by crossing forces with x_cp

        C_Y = self.C_Yb * self.beta #side force
        C_l = self.C_l + self.C_lr * r_hat + self.C_lp * p_hat #roll
        C_n = self.C_np * p_hat + self.C_nr * r_hat #+ self.C_nb * self.beta#yaw force

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

        body_forces_wind = np.array([-body_drag, body_side, -body_lift],'d')
        body_forces_body = quat.rotateVectorQ(wind_to_body, body_forces_wind)

        aero_moments = np.array([body_rolling_moment, body_pitching_moment, body_yawing_moment],'d')

        moments_with_torque = np.array([
            aero_moments[0] - self.cp_wrt_cm[2]*body_forces_body[1] + self.cp_wrt_cm[1]*body_forces_body[2],
            aero_moments[1] + self.cp_wrt_cm[2]*body_forces_body[0] - self.cp_wrt_cm[0]*body_forces_body[2],
            aero_moments[2] - self.cp_wrt_cm[1]*body_forces_body[0] + self.cp_wrt_cm[0]*body_forces_body[1],
        ], 'd')

        if self.has_gridfins == 1:
            gridfin_forces, gridfin_moments = self.calculate_grid_fin_forces()
            body_forces_body = body_forces_body + gridfin_forces
            moments_with_torque = moments_with_torque + gridfin_moments

        if self.on_balloon == 1:
            balloon_forces, moment_from_balloon = self.get_buoyancy_force()
            body_forces_body = body_forces_body + balloon_forces
            moments_with_torque = moments_with_torque + moment_from_balloon


        return body_forces_body, moments_with_torque

    def get_control_deflection(self):
        """Returns the current control state"""
        return np.array([(self.rdr + self.trim_rdr), \
                        (self.ail + self.trim_ail), \
                        (self.el + self.trim_el), \
                        (self.power + self.trim_power)], 'd')

    def use_realistic_mixing(self):
        """Realistic mixing, where surfaces are mapped to aileron, elevator, throttle"""
        self.plav_mixing = 0

    def use_plav_mixing(self):
        """PLAV Mixing"""
        self.plav_mixing = 1

    def trigger_event(self):
        """Triggers cut_balloon event"""
        self.cut_balloon()

    def cut_balloon(self):
        """Cuts the balloon"""
        self.on_balloon = 0
        self.Sref = self.brgr_Sref
        self.mass = self.brgr_mass

    def get_buoyancy_force(self):
        """Solves for buoyancy forces"""
        self.mass = self.brgr_mass + 3.5 #3kg balloon + 0.5 kg He

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
            self.cut_balloon()
            return np.zeros(3,'d'), np.zeros(3,'d')

        bouancy_force = self.density * self.balloon_volume * -self.gravity

        mounting_arm = np.array([-0.4826, 0.001, 0.0254], 'd') #from cm to center of balloon
        bouancy_moment = np.cross(mounting_arm, bouancy_force)

        return bouancy_force, bouancy_moment

    def calculate_grid_fin_forces(self):
        """Calculates the forces of each grid fin"""
        qbar = self.get_qbar()
        S = self.Sref

        #there's an issue with the reference frame
        #let's calculate the forces in the wind frame
        #and then rotate them to the body frame

        #gamma describes the position on tube around +X (fwd), 0 is starboard
        gamma_top  = -math.pi/2.0
        gamma_star =  math.pi/6.0
        gamma_port =  5.0*math.pi/6.0

        #control mixing
        yaw_adjustment_factor = 0.5
        ail_command = self.ail + self.trim_ail
        el_command  = self.el + self.trim_el
        rdr_command = self.rdr + self.trim_rdr

        if self.plav_mixing == 1:
            #theta is the angle of deflection of the surface
            deflection_top_command  = -ail_command*0.2 + rdr_command*0.2
            deflection_star_command = -ail_command*0.2 + rdr_command*yaw_adjustment_factor*0.2 + el_command*0.5
            deflection_port_command = -ail_command*0.2 + rdr_command*yaw_adjustment_factor*0.2 - el_command*0.5
        else:
            deflection_top_command = self.ail    * math.pi/2
            deflection_star_command = self.el     * math.pi/2
            deflection_port_command = self.power  * math.pi/2

        #this is a low pass filter to make the servos act realistic
        b_0 = 0.0154662914
        b_1 = 0.0154662914
        a_1 = 0.9690674172
        current_command = np.array([deflection_top_command, deflection_star_command, deflection_port_command, self.power], 'd')

        deflection_top = b_0 * current_command[0] + b_1 * self.prev_command[0] + a_1 * self.prev_position[0]
        deflection_star = b_0 * current_command[1] + b_1 * self.prev_command[1] + a_1 * self.prev_position[1]
        deflection_port = b_0 * current_command[2] + b_1 * self.prev_command[2] + a_1 * self.prev_position[2]

        #store previous commands and positions
        self.prev_command = current_command
        self.prev_position = np.array([deflection_top, deflection_star, deflection_port], 'd')

        #get local alpha and beta in to each fin
        ab_top  = get_local_alpha_beta(self.velocity, gamma_top,  deflection_top )
        ab_star = get_local_alpha_beta(self.velocity, gamma_star, deflection_star)
        ab_port = get_local_alpha_beta(self.velocity, gamma_port, deflection_port)

        #get the drag of the fin (force along fin chord)
        drag_angle_top  = np.sqrt(ab_top[0]**2  + ab_top[1]**2 )
        drag_angle_star = np.sqrt(ab_star[0]**2 + ab_star[1]**2)
        drag_angle_port = np.sqrt(ab_port[0]**2 + ab_port[1]**2)

        drag_top  = np.interp(drag_angle_top,  self.C_XYlutX, self.C_XlutY)
        drag_star = np.interp(drag_angle_star, self.C_XYlutX, self.C_XlutY)
        drag_port = np.interp(drag_angle_port, self.C_XYlutX, self.C_XlutY)

        #get the normal and radial forces
        #normal is perpendicular to chord and radial, radial inline with hinge
        normal_top  = np.interp(ab_top[0],  self.C_XYlutX, self.C_YlutY)
        radial_top  = np.interp(ab_top[1],  self.C_XYlutX, self.C_YlutY) * 0.5
        normal_star = np.interp(ab_star[0], self.C_XYlutX, self.C_YlutY)
        radial_star = np.interp(ab_star[1], self.C_XYlutX, self.C_YlutY) * 0.5
        normal_port = np.interp(ab_port[0], self.C_XYlutX, self.C_YlutY)
        radial_port = np.interp(ab_port[1], self.C_XYlutX, self.C_YlutY) * 0.5

        #get the rotation matrix for each to rotate around +X
        position_rot_top  = get_x_rotation_matrix(gamma_top )
        position_rot_star = get_x_rotation_matrix(gamma_star)
        position_rot_port = get_x_rotation_matrix(gamma_port)

        #coeff forces in local wind frame of grid fin
        coeff_forces_wind_top  = np.array([-drag_top,  -radial_top, -normal_top],'d')
        coeff_forces_wind_star = np.array([-drag_star, -radial_star,-normal_star],'d')
        coeff_forces_wind_port = np.array([-drag_port, -radial_port,-normal_port],'d')

        #from coeff to real force in local wind
        forces_wind_top  = qbar * S * coeff_forces_wind_top
        forces_wind_star = qbar * S * coeff_forces_wind_star
        forces_wind_port = qbar * S * coeff_forces_wind_port

        #rotate forces to body frame

        #quaternion from local wind frame to hinge frame (the ptfe bearing)
        wind_to_body_top  = get_wind_to_body_axis(ab_top[0],  ab_top[1])
        wind_to_body_star = get_wind_to_body_axis(ab_star[0], ab_star[1])
        wind_to_body_port = get_wind_to_body_axis(ab_port[0], ab_port[1])

        #rotate forces to hinge frame
        hinge_forces_top = quat.rotateVectorQ(wind_to_body_top, forces_wind_top)
        hinge_forces_star = quat.rotateVectorQ(wind_to_body_star, forces_wind_star)
        hinge_forces_port = quat.rotateVectorQ(wind_to_body_port, forces_wind_port)

        #quaternion from hinge frame to body frame (wow what we want)
        body_force_top  = position_rot_top  @ hinge_forces_top
        body_force_star = position_rot_star @ hinge_forces_star
        body_force_port = position_rot_port @ hinge_forces_port

        #how far the arm is from the hinge to the center of pressure [m]
        grid_fin_arm = np.array([-0.5, 0.0889, 0.0], 'd')

        arm_top  = position_rot_top  @ grid_fin_arm #rotates the arm for each fin
        arm_star = position_rot_star @ grid_fin_arm
        arm_port = position_rot_port @ grid_fin_arm

        #grid fins have tiny moments, torques come from the lever arm
        moment_top  = np.cross(arm_top,  body_force_top )
        moment_star = np.cross(arm_star, body_force_star)
        moment_port = np.cross(arm_port, body_force_port)

        total_force  = body_force_top + body_force_star + body_force_port
        total_moment = moment_top + moment_star + moment_port

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

def init_aircraft(config_file) -> BRGRConfig:
    """Init aircraft from json file"""
    mass = config_file['mass']
    inertia = np.array(config_file['inertiatensor'])
    cmac = config_file['cref']
    Sref = config_file['Sref']
    bref = config_file['bref']
    C_L0 = config_file['C_L0']
    C_La = config_file['C_La']
    C_Lmax = config_file['C_Lmax']
    C_Lmin = config_file['C_Lmin']
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

    trim_rudder   = config_file['trim_rudder']
    trim_aileron  = config_file['trim_aileron']
    trim_elevator = config_file['trim_elevator']
    trim_throttle = config_file['trim_throttle']

    C_XYlutX = np.array(config_file['C_XYlutX'],'d')
    C_XlutY  = np.array(config_file['C_XlutY'], 'd')
    C_YlutY  = np.array(config_file['C_YlutY'], 'd')

    if config_file['plav_mixing']:
        plav_mixing = 1
        print("Using PLAV mixing")
    else:
        plav_mixing = 0
        print("Using realistics mixing")

    if config_file['on_balloon']:
        on_balloon = 1
        gas_cf = config_file['gas_cf']
        burst_dia_ft = config_file['burst_dia_ft']
        print("Using balloon model")
    else:
        on_balloon = 0
        gas_cf = 0.0
        burst_dia_ft = 0.0

    aircraft_model = BRGRConfig(mass, inertia, cmac, Sref, bref, cp_wrt_cm,\
                                C_L0, C_La, C_Lmax, C_Lmin, C_D0, epsilon, C_m0, C_ma, C_mq,\
                                C_Yb, C_l, C_lp, C_lr, C_np, C_nr, C_mbb, C_Db,\
                                C_nb, trim_rudder, trim_aileron, trim_elevator, trim_throttle,
                                1, C_XYlutX, C_XlutY, C_YlutY, plav_mixing, on_balloon, gas_cf, burst_dia_ft)
    #none for control unit
    return aircraft_model

#modelparam = load_scenario("scenarios/brgrDroneDrop.json")

#air = init_aircraft(modelparam)
