"""2-stage rocket model from NASA-TM-2015-218675
implements aero, inertia, and prop"""
import math
import sys


import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

from plav.vehicle_models.generic_aircraft_config import get_dynamic_viscosity, velocity_to_alpha_beta, get_wind_to_body_axis
from plav.vehicle_models.generic_aircraft_config import AircraftConfig
from plav.quaternion_math import rotateVectorQ

spec = [

    #geometrics
    ('mass', float64),
    ('cmac', float64),
    ('Sref', float64),
    ('bref', float64),
    ('inertiamatrix', float64[:,::1]),
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

    #state
    ('stage1firing_flag', int64),
    ('stage2firing_flag', int64),
    ('rocketHasStaged', int64),
    ('mass_dot', float64),
    ('stage1fuelConsumed', float64),
    ('stage2fuelConsumed', float64),
    ('vrsPositionOfMrc', float64),

    ('last_time', float64),
    ('time', float64),
    ('stage1brunout_time', float64)
]

@jit(float64(float64, float64, float64[:], float64[:], float64[:,:]))
def bilinear_interp(x, y, x_grid, y_grid, z_grid):
    """Billinear interpolation with flat extrapolation
    https://en.wikipedia.org/wiki/Bilinear_interpolation"""
    if x < x_grid[0]:
        x = x_grid[0]
    if x > x_grid[-1]:
        x = x_grid[-1]
    if y < y_grid[0]:
        y = y_grid[0]
    if y > y_grid[-1]:
        y = y_grid[-1]

    x_index = np.searchsorted(x_grid, x) -1
    y_index = np.searchsorted(y_grid, y) -1

    x1 = x_grid[x_index]
    x2 = x_grid[x_index+1]
    y1 = y_grid[y_index]
    y2 = y_grid[y_index+1]

    q11 = z_grid[y_index,   x_index  ]
    q12 = z_grid[y_index+1, x_index  ]
    q21 = z_grid[y_index,   x_index+1]
    q22 = z_grid[y_index+1, x_index+1]
    
    denom = (x2-x1)*(y2-y1)
    numer = (q11*(x2-x)*(y2-y) + q12*(x2-x)*(y-y1) + q21*(x-x1)*(y2-y) + q22*(x-x1)*(y-y1))

    return numer/denom

def init_aircraft(modelparam) -> AircraftConfig:
    """Initalizes F16 aircraft maybe with HITL"""
    aircraft = TwoStageRocket()

    return aircraft

@jitclass(spec)
class TwoStageRocket(object):
    """Object used to lookup coefficients for 2 stage rocket"""
    def __init__(self):


        self.mass = 314000.
        self.cmac = 3.
        self.Sref = 3.
        self.bref = 7.

        self.inertiamatrix = np.array([
            [353250, 0, 0.0 ],
            [0, 33501637.473461, 0],
            [0.0, 0, 33501637.473461]
        ], 'd') #kg m^2

        self.altitude = 0.0
        self.velocity = np.zeros(3,'d')
        self.omega = np.zeros(3,'d')
        self.airspeed = 0
        self.alpha = 0
        self.beta = 0
        self.reynolds = 0
        self.density = 0
        self.temperature = 0
        self.mach = 0
        self.gravity = np.zeros(3,'d')
        self.time = 0.0

        self.cp_wrt_cm = np.zeros(3,'d')

        self.stage1firing_flag = 1
        self.stage2firing_flag = 0
        self.rocketHasStaged = 0
        self.mass_dot = 0.0
        self.stage1fuelConsumed = 0.0
        self.stage2fuelConsumed = 0.0
        self.vrsPositionOfMrc = 16.918790
        self.last_time = 0.0
        self.stage1burnout_time = 0.0


    def update_control(self, rudder, aileron, elevator, throttle):
        """Not implemented for this model"""
        return

    def update_trim(self):
        """Not implemented for this model"""
        return
    
    def update_conditions(self, altitude, velocity, omega, density, temperature, speed_of_sound, gravity, time):
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

        self.time = time

        self.update_vehicle_state()

    def update_vehicle_state(self):
        """Updates the vehicle state regarding mass and staging"""
        fullVehicleMomentReferenceCenter = 16.918790
        secondStageMomentReferenceCenter = 4.797980

        Liftoff_XCG = 16.91879
        S1Burnout_XCG = 9.421642
        S2Ignite_XCG = 4.79798
        S2Burnout_XCG = 3.947368

        Liftoff_Mass = 314000.
        S1Burnout_Mass = 134000.
        S2Ignite_Mass = 99000.
        S2Burnout_Mass= 19000.
    
        Liftoff_Ixx = 353250.
        S1Burnout_Ixx = 150750.
        S2Ignite_Ixx = 111375.
        S2Burnout_Ixx = 21375.
        Liftoff_Iyy = 33501637.473461
        S1Burnout_Iyy = 10886636.572139
        S2Ignite_Iyy = 941063.762626
        S2Burnout_Iyy = 212384.868421

        stage1XCGRange = Liftoff_XCG - S1Burnout_XCG
        stage2XCGRange = S2Ignite_XCG - S2Burnout_XCG
        stage1FuelCapacity = Liftoff_Mass - S1Burnout_Mass
        stage2FuelCapacity = S2Ignite_Mass - S2Burnout_Mass

        stage1IxxRange = Liftoff_Ixx - S1Burnout_Ixx
        stage2IxxRange = S2Ignite_Ixx - S2Burnout_Ixx
        stage1IyyRange = Liftoff_Iyy - S1Burnout_Iyy
        stage2IyyRange = S2Ignite_Iyy - S2Burnout_Iyy

        stage1FuelRemainingFrac = 1.0 - (self.stage1fuelConsumed / stage1FuelCapacity)
        stage2FuelRemainingFrac = 1.0 - (self.stage2fuelConsumed / stage2FuelCapacity)

        if self.rocketHasStaged > 0:
            self.vrsPositionOfCM_X = S2Burnout_XCG + stage2XCGRange * (self.stage2fuelConsumed/stage2FuelCapacity)
            self.mass = S2Burnout_Mass + stage2FuelCapacity * stage2FuelRemainingFrac
            self.inertiamatrix[0,0] = S2Burnout_Ixx + stage2IxxRange * stage2FuelRemainingFrac
            self.inertiamatrix[1,1] = S2Burnout_Iyy + stage2IyyRange * stage2FuelRemainingFrac
            self.inertiamatrix[2,2] = self.inertiamatrix[1,1]
            self.vrsPositionOfMrc = secondStageMomentReferenceCenter
        else:
            self.mass = S1Burnout_Mass + stage1FuelCapacity * stage1FuelRemainingFrac
            self.inertiamatrix[0,0] = S1Burnout_Ixx + stage1IxxRange * stage1FuelRemainingFrac
            self.inertiamatrix[1,1] = S1Burnout_Iyy + stage1IyyRange * stage1FuelRemainingFrac
            self.inertiamatrix[2,2] = self.inertiamatrix[1,1]
            self.vrsPositionOfCM_X = S1Burnout_XCG + stage1XCGRange * (self.stage1fuelConsumed/stage1FuelCapacity)
            self.vrsPositionOfMrc = fullVehicleMomentReferenceCenter
        
        self.cp_wrt_cm[0] =  self.vrsPositionOfCM_X - self.vrsPositionOfMrc

        if self.stage1firing_flag > 0:
            self.stage1fuelConsumed += self.mass_dot * (self.time - self.last_time)
            if self.stage1fuelConsumed >= stage1FuelCapacity:
                self.stage1fuelConsumed = stage1FuelCapacity
                self.stage1firing_flag = 0
                self.stage1burnout_time = self.time

        if self.time - self.stage1burnout_time > 96.79:
            self.trigger_event()
            
        if self.stage2firing_flag > 0:
            self.stage2fuelConsumed += self.mass_dot * (self.time - self.last_time)
            if self.stage2fuelConsumed >= stage2FuelCapacity:
                self.stage2fuelConsumed = stage2FuelCapacity
                self.stage2firing_flag = 0
        
        self.last_time = self.time

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

        #moments_with_torque = np.array([
        #    aero_moments[0] - self.cp_wrt_cm[2]*body_forces_body[1] + self.cp_wrt_cm[1]*body_forces_body[2],
        #    aero_moments[1] + self.cp_wrt_cm[2]*body_forces_body[0] - self.cp_wrt_cm[0]*body_forces_body[2],
        #    aero_moments[2] - self.cp_wrt_cm[1]*body_forces_body[0] + self.cp_wrt_cm[0]*body_forces_body[1],
        #], 'd')

        return body_forces_body, np.zeros(3,'d')#aero_moments

    def get_coeff(self):
        """Gets the aerodynamic coefficients of the 2 stage rocket
        alpha and beta are in degrees"""


        alpha = np.rad2deg(self.alpha) #object alpha is rad, this one is degrees
        beta = np.rad2deg(self.beta)
        #as defined in twostage_aero.dml
        alpha_total = math.sqrt(alpha**2 + beta**2)

        C_L = cl_lookup(alpha_total)
        C_D = cd_lookup(alpha_total)
        C_m = cm_lookup(alpha)
        C_Y = cy_lookup(beta)
        C_l = 0.0
        C_n = cn_lookup(beta)

        return C_L,C_D,C_m, C_Y, C_l, C_n
    
    def calculate_thrust(self):
        """Calculates the thrust
        """
        
        stage1maxThrust = 17000000. #N
        stage2maxThrust = 5000000. #N
        stage1isp = 360.
        stage2isp = 390.
        thrust = 0.0
        isp = 0.001

        if self.stage1firing_flag > 0:
            isp = stage1isp
        if self.stage2firing_flag > 0:
            isp = stage2isp

        if self.stage1firing_flag > 0:
            thrust = stage1maxThrust
        if self.stage2firing_flag > 0:
            thrust = stage2maxThrust

        if thrust == 0:
            self.mass_dot = 0.0
        else:
            self.mass_dot = thrust/(isp*9.8066) #kg/s

        return thrust

    def get_control_command(self):
        """Not implemented"""
        return np.zeros(4,'d')

    def get_control_deflection(self):
        """Not implemented"""
        return np.zeros(4,'d')

    def trigger_event(self):
        """Triggers staging event"""
        if self.rocketHasStaged == 0:
            self.rocketHasStaged = 1
            self.stage1firing_flag = 0
            self.stage2firing_flag = 1

    def get_xcp(self):
        """returns x_cp with respect to CM"""
        return self.cp_wrt_cm

    def get_inertia_matrix(self):
        """Returns inertia matrix as 2d np array"""
        return self.inertiamatrix

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

    def input_control(self, elevator, aileron, rudder):
        """Not implemented"""
        return

    def get_control_deflection(self):
        """Not implemented"""
        return np.zeros(4,'d')

@jit(float64[:]())
def get_alpha_table_forces():
    """Get Alpha for force lookups"""
    alpha_table = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.], 'd')
    return alpha_table

@jit(float64[:]())
def get_alpha_table_moments():
    """Get Alpha for moment lookups"""
    alpha_table = np.array([-20., 0., 20.], 'd')
    return alpha_table

@jit(float64[:]())
def get_beta_table_forces():
    """Get Beta for force lookups"""
    beta_table = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.], 'd')
    return beta_table

@jit(float64[:]())
def get_beta_table_moments():
    """Get Beta for moment lookups"""
    beta_table = np.array([-20., 0., 20.], 'd')
    return beta_table

@jit(float64(float64))
def cl_lookup(alpha):
    """Lookup for C_L, based on Alpha [deg]"""
    alpha_table = get_alpha_table_forces()
    table = np.array([-1.6, -1.0, -0.73, -0.49, -0.24, 0, 0.24, 0.49, 0.73, 1.0, 1.6], 'd')
    C_L = np.interp(alpha, alpha_table, table)
    return C_L

@jit(float64(float64))
def cy_lookup(beta):
    """Lookup for C_Y (side force), based on Beta [deg]"""
    beta_table = get_beta_table_forces()
    table = np.array([1.6, 1.0, 0.73, 0.49, 0.24, 0, -0.24, -0.49, -0.73, -1.0, -1.6], 'd')
    C_Y = np.interp(beta, beta_table, table)
    return C_Y

@jit(float64(float64))
def cd_lookup(alpha):
    """Lookup for C_D, based on Total Alpha [deg]"""
    alpha_table = get_alpha_table_forces()
    table = np.array([0.48, 0.38, 0.31, 0.25, 0.23, 0.21, 0.23, 0.25, 0.31, 0.38, 0.48], 'd')
    C_D = np.interp(alpha, alpha_table, table)
    return C_D

@jit(float64(float64))
def cm_lookup(alpha):
    """Lookup for C_m (pitching moment), based on Alpha [deg]"""
    alpha_table = get_alpha_table_moments()
    table = np.array([0.6,   0,  -0.6], 'd')
    C_m = np.interp(alpha, alpha_table, table)
    return C_m

@jit(float64(float64))
def cn_lookup(beta):
    """Lookup for C_n (yaw moment), based on beta [deg]"""
    beta_table = get_beta_table_moments()
    table = np.array([0.6,   0,  -0.6], 'd')
    C_n = np.interp(beta, beta_table, table)
    return C_n
