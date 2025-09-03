"""One-line log for extracting certain aircraft properties"""

import math
import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

from plav.quaternion_math import to_euler
import plav.conversions as conv

#index of data in sim_data
SDI_TIME = 0
SDI_Q1 = 1
SDI_Q2 = 2
SDI_Q3 = 3
SDI_Q4 = 4
SDI_P = 5
SDI_Q = 6
SDI_R = 7
SDI_LONG = 8
SDI_LAT = 9
SDI_ALT = 10
SDI_VN = 11
SDI_VE = 12
SDI_VD = 13
SDI_ROLL = 14
SDI_PITCH = 15
SDI_YAW = 16
SDI_FX = 17
SDI_FY = 18
SDI_FZ = 19
SDI_MX = 20
SDI_MY = 21
SDI_MZ = 22
SDI_GRAVITY = 23
SDI_SOUND = 24
SDI_MACH = 25
SDI_QBAR = 26
SDI_AIR_DENSITY = 27
SDI_AIR_PRESSURE = 28
SDI_AIR_TEMPERATURE = 29
SDI_TAS = 30
SDI_ALPHA = 31
SDI_BETA = 32
SDI_REYNOLDS = 33
SDI_FLIGHT_PATH = 34
SDI_DOWNRANGE = 35
SDI_THRUST = 36
SDI_AILERON_CMD = 37
SDI_ELEVATOR_CMD = 38
SDI_RUDDER_CMD = 39
SDI_THRUST_CMD = 40
SDI_DELTA_N = 41
SDI_DELTA_E = 42
SDI_DELTA_D = 43
SDI_AX = 44
SDI_AY = 45
SDI_AZ = 46
SDI_WIND_N = 47
SDI_WIND_E = 48
SDI_WIND_D = 49
SDI_CONTROL_0 = 50
SDI_CONTROL_1 = 51
SDI_CONTROL_2 = 52
SDI_CONTROL_3 = 53

SDI_LINE_SIZE = 54 #total size of a line

spec = [

    ('time', float64),

    #positionals
    ('lon_lat_alt', float64[:]),
    ('ned_velocity', float64[:]),

    ('body_acceleration', float64[:]),

    #rotationals
    ('body_rate', float64[:]),
    ('quat', float64[:]),
    ('euler', float64[:]),

    #forces and moments
    ('aero_body_force', float64[:]),
    ('aero_body_moment', float64[:]),
    ('thrust', float64),

    #enviroment
    ('local_gravity', float64),
    ('speed_of_sound', float64),
    ('mach', float64),
    ('dynamic_pressure', float64),
    ('true_airspeed', float64),
    ('air_density', float64),
    ('ambient_pressure', float64),
    ('ambient_temperature', float64),

    ('control_deflection', float64[:]),
    ('control_command', float64[:]),
    ('wind_ned', float64[:]),

    #derived
    ('alpha', float64),
    ('beta', float64),
    ('reynolds', float64),

    ('data', float64[:,:]),
    ('data_columns', int64),
    ('valid_data_size', int64),
    ('line', float64[:]),
    ('save_interval', float64),

]

@jitclass(spec)
class SimDataLogger(object):
    """Jitted Logger Object to Run in the Function as an arg"""

    def __init__(self, preallocated = 1):
        self.save_interval = 0.1 #configurable saving interval

        self.time = 0.0

        self.lon_lat_alt = np.zeros(3)
        self.ned_velocity = np.zeros(3)
        self.body_rate = np.zeros(3)
        self.quat = np.zeros(4)
        self.euler = np.zeros(3)

        self.aero_body_force = np.zeros(3)
        self.aero_body_moment = np.zeros(3)

        self.local_gravity = 0.0
        self.speed_of_sound = 0.0

        self.mach = 0.0
        self.dynamic_pressure = 0.0
        self.true_airspeed = 0.0

        self.air_density = 0.0
        self.ambient_pressure = 0.0
        self.ambient_temperature = 0.0

        self.alpha = 0.0
        self.beta = 0.0
        self.reynolds = 0.0
        self.thrust = 0.0

        self.wind_ned = np.zeros(3,'d')

        self.control_deflection = np.zeros(4,'d')
        self.control_command = np.zeros(4,'d')
        self.body_acceleration = np.zeros(3,'d')

        self.line = self.make_line()

        self.data_columns = int64(np.size(self.line))
        self.data = np.zeros((self.data_columns, int64(preallocated)))
        self.valid_data_size = 0

    def get_lastest(self):
        """Returns the last line of data"""
        if self.valid_data_size == 0:
            return None
        else:
            return self.line

    def load_line(self, time, state, aero_body_force, \
                    aero_body_moment, local_gravity, speed_of_sound, mach ,dynamic_pressure, \
                    true_airspeed, air_density, ambient_pressure, ambient_temperature, \
                    alpha, beta, reynolds, thrust, control_deflection, body_acceleration,\
                    wind_ned, control_command):
        """Loads a line of data for the object so it can be used for the logger"""

        self.time = np.array([time])
        self.quat         = state[0:4]
        self.body_rate    = state[4:7]
        self.lon_lat_alt  = state[7:10]
        self.ned_velocity = state[10:13]

        self.aero_body_force  = aero_body_force
        self.aero_body_moment = aero_body_moment

        self.local_gravity  = np.array([local_gravity])
        self.speed_of_sound = np.array([speed_of_sound])

        self.mach = np.array([mach])
        self.dynamic_pressure = np.array([dynamic_pressure])
        self.true_airspeed = np.array([true_airspeed])

        self.air_density = np.array([air_density])
        self.ambient_pressure = np.array([ambient_pressure])
        self.ambient_temperature = np.array([ambient_temperature])

        self.alpha = np.array([alpha])
        self.beta = np.array([beta])
        self.reynolds = np.array([reynolds])
        self.thrust = np.array([thrust], 'd')
        self.control_deflection = control_deflection
        self.body_acceleration = body_acceleration
        self.wind_ned = wind_ned
        self.control_command = control_command

    def make_line(self):
        """Makes a line of data"""

        rollpitchyaw = to_euler(self.quat)
        flight_path = calculate_flight_path_angle(self.ned_velocity)

        if self.valid_data_size != 0:
            inital_lat = self.data[SDI_LAT][0]
            inital_lon = self.data[SDI_LONG][0]
            inital_alt = self.data[SDI_ALT][0]
            downrange = dist_vincenty(self.lon_lat_alt[1],self.lon_lat_alt[0],inital_lat,inital_lon)
            ned_traveled = ned_displacement(inital_lat, inital_lon, inital_alt,
                                            self.lon_lat_alt[1],self.lon_lat_alt[0],self.lon_lat_alt[2])
        else:
            downrange = 0.0
            ned_traveled = np.zeros(3,'d')

        line = np.zeros(SDI_LINE_SIZE, 'd')
        line[SDI_TIME] = self.time
        line[SDI_Q1] = self.quat[0]
        line[SDI_Q2] = self.quat[1]
        line[SDI_Q3] = self.quat[2]
        line[SDI_Q4] = self.quat[3]
        line[SDI_P] = self.body_rate[0]
        line[SDI_Q] = self.body_rate[1]
        line[SDI_R] = self.body_rate[2]
        line[SDI_LONG] = self.lon_lat_alt[0]
        line[SDI_LAT] = self.lon_lat_alt[1]
        line[SDI_ALT] = self.lon_lat_alt[2]
        line[SDI_VN] = self.ned_velocity[0]
        line[SDI_VE] = self.ned_velocity[1]
        line[SDI_VD] = self.ned_velocity[2]
        line[SDI_ROLL] = rollpitchyaw[0]
        line[SDI_PITCH] = rollpitchyaw[1]
        line[SDI_YAW] = rollpitchyaw[2]
        line[SDI_FX] = self.aero_body_force[0]
        line[SDI_FY] = self.aero_body_force[1]
        line[SDI_FZ] = self.aero_body_force[2]
        line[SDI_MX] = self.aero_body_moment[0]
        line[SDI_MY] = self.aero_body_moment[1]
        line[SDI_MZ] = self.aero_body_moment[2]
        line[SDI_GRAVITY] = self.local_gravity
        line[SDI_SOUND] = self.speed_of_sound
        line[SDI_MACH] = self.mach
        line[SDI_QBAR] = self.dynamic_pressure
        line[SDI_AIR_DENSITY] = self.air_density
        line[SDI_AIR_PRESSURE] = self.ambient_pressure
        line[SDI_AIR_TEMPERATURE] = self.ambient_temperature
        line[SDI_TAS] = self.true_airspeed
        line[SDI_ALPHA] = self.alpha
        line[SDI_BETA] = self.beta
        line[SDI_REYNOLDS] = self.reynolds
        line[SDI_FLIGHT_PATH] = flight_path
        line[SDI_DOWNRANGE] = downrange
        line[SDI_THRUST] = self.thrust
        line[SDI_RUDDER_CMD] = self.control_deflection[0]
        line[SDI_AILERON_CMD] = self.control_deflection[1]
        line[SDI_ELEVATOR_CMD] = self.control_deflection[2]
        line[SDI_THRUST_CMD] = self.control_deflection[3]
        line[SDI_DELTA_N] = ned_traveled[0]
        line[SDI_DELTA_E] = ned_traveled[1]
        line[SDI_DELTA_D] = ned_traveled[2]
        line[SDI_AX] = self.body_acceleration[0]
        line[SDI_AY] = self.body_acceleration[1]
        line[SDI_AZ] = self.body_acceleration[2]
        line[SDI_WIND_N] = self.wind_ned[0]
        line[SDI_WIND_E] = self.wind_ned[1]
        line[SDI_WIND_D] = self.wind_ned[2]
        line[SDI_CONTROL_0] = self.control_command[0]
        line[SDI_CONTROL_1] = self.control_command[1]
        line[SDI_CONTROL_2] = self.control_command[2]
        line[SDI_CONTROL_3] = self.control_command[3]

        return line

    def save_line(self):
        """Saves the currently loaded data as a line"""
        self.line = self.make_line()

        this_time = self.line[SDI_TIME]
        last_time = self.data[SDI_TIME, self.valid_data_size - 1]

        if self.valid_data_size == 0 or (this_time - last_time) >= self.save_interval:
            self.append_data(self.line)

    def increase_size(self):
        """Double the data size if necessary"""
        self.data = np.append(self.data, self.data, axis=1) #np.pad() isn't implemented in numba lol

    def append_data(self, new_line):
        """Append a new line of data"""          
        if self.valid_data_size + 1 > np.size(self.data) // self.data_columns:
            self.increase_size()

        self.data[:, self.valid_data_size] = new_line

        self.valid_data_size = self.valid_data_size + 1

    def actual_data_size(self):
        return np.shape(self.data)

    def trim_excess(self):
        """Trim excess data
        excess points are the preallocated zeros"""
        new_data = self.data[:self.data_columns, :self.valid_data_size]
        self.data = new_data

    def return_data(self):
        """Returns the whole data array"""
        self.trim_excess()
        return self.data

    def return_data_size(self):
        """Returns the size of data"""
        self.trim_excess()
        return self.valid_data_size

#a bunch of helper functions to record other data
@jit#(float64(float64[:]))
def calculate_flight_path_angle(velocity_ned):
    """
    Calculates the flight path angle [rad] of the aircraft given the velocity in NED

    Parameters
    ----------
    velocity_NED : array_like
        The velocity in north-east-down coordinates
        Units irrrelevant

    Returns
    -------
    
    flight_path_angle : float
        The flight path angles in radians
    """
    ground_speed = math.sqrt(velocity_ned[0]**2 + velocity_ned[1] **2)

    flight_path_angle = math.atan2(velocity_ned[2], ground_speed)

    return np.float64(flight_path_angle)

@jit(float64[:](float64, float64, float64))
def lla_to_ecef(lat, lon, h_m):
    """Geodetic (lat, lon, h) -> ECEF (x, y, z) in meters, WGS-84."""

    _A = 6378137.0                      # semi-major axis [m]
    _F = 1 / 298.257223563              # flattening
    _E2 = _F * (2 - _F)                 # eccentricity^2


    s = math.sin(lat)
    c = math.cos(lat)
    N = _A / math.sqrt(1.0 - _E2 * s * s)  # prime vertical radius of curvature

    x = (N + h_m) * c * math.cos(lon)
    y = (N + h_m) * c * math.sin(lon)
    z = (N * (1.0 - _E2) + h_m) * s
    return np.array([x, y, z], 'd')

@jit(float64[:,:](float64, float64))
def ecef_to_ned_matrix(lat, lon):
    """
    Rotation matrix R that maps ECEF vectors to NED at (lat, lon).
    ned = R @ vec_ecef
    """

    sL, cL = math.sin(lat), math.cos(lat)
    s_lambda, c_lambda = math.sin(lon), math.cos(lon)

    return np.array([
        [-sL * c_lambda, -sL * s_lambda,  cL],
        [   -s_lambda   ,     c_lambda   ,  0.0],
        [-cL * c_lambda, -cL * s_lambda, -sL],
    ],'d')

@jit(float64[:](float64, float64, float64, float64, float64, float64))
def ned_displacement(lat0, lon0, h0, lat1, lon1, h1):
    """
    NED displacement (meters) from (lat0,lon0,h0) to (lat1,lon1,h1),
    defined at the initial point's local tangent plane.
    """
    r0 = lla_to_ecef(lat0, lon0, h0)
    r1 = lla_to_ecef(lat1, lon1, h1)
    dr_ecef = r1 - r0                   # chord vector in ECEF
    R_en = ecef_to_ned_matrix(lat0, lon0)
    ned = R_en @ dr_ecef
    ned[2] = h0 - h1
    return ned

@jit(float64(float64, float64, float64, float64))
def dist_vincenty(lat1, lon1, lat2, lon2):
    """distance with Vincenty formula."""

    max_iter=20
    tol=1e-12

    # WGS‑84 ellipsoid parameters
    a = 6378137.0               # semi-major axis (meters)
    f = 1 / 298.257223563       # flattening
    b = a * (1 - f)             # semi-minor axis

    phi1, phi2 = lat1, lat2
    l = lon2 - lon1

    u1 = math.atan((1 - f) * math.tan(phi1))
    u2 = math.atan((1 - f) * math.tan(phi2))
    sin_u1, cos_u1 = math.sin(u1), math.cos(u1)
    sin_u2, cos_u2 = math.sin(u2), math.cos(u2)

    lamb = l
    for _ in range(max_iter):
        sin_lambda, cos_lambda = math.sin(lamb), math.cos(lamb)
        sin_sigma = math.hypot(cos_u2 * sin_lambda,
                          cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda)
        if sin_sigma == 0:
            return 0.0  # coincident points

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma
        cos2_alpha = 1 - sin_alpha * sin_alpha

        cos2_sigmam = (cos_sigma - 2 * sin_u1 * sin_u2 / cos2_alpha) if cos2_alpha != 0 else 0.0

        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))

        lamb_prev = lamb
        lamb = (l + (1 - C) * f * sin_alpha *
                (sigma + C * sin_sigma * (cos2_sigmam + C * cos_sigma *
                 (-1 + 2 * cos2_sigmam * cos2_sigmam))))
        if abs(lamb - lamb_prev) < tol:
            break

    # handle non-convergence: proceed with best estimate anyway
    u2 = cos2_alpha * (a*a - b*b) / (b*b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175*u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47*u2)))
    delta_sigma = (B * sin_sigma * (cos2_sigmam + B / 4 * (
              cos_sigma * (-1 + 2 * cos2_sigmam*cos2_sigmam) -
              B / 6 * cos2_sigmam * (-3 + 4*sin_sigma*sin_sigma) *
              (-3 + 4*cos2_sigmam*cos2_sigmam))))
    s = b * A * (sigma - delta_sigma)

    return np.float64(s)

def return_data_for_csv(sim_data) -> dict:
        """Returns the data in a format suitable for CSV export"""

        csv_data = {'time': sim_data[SDI_TIME],
        'altitudeMsl_ft': sim_data[SDI_ALT]*conv.M_TO_FT,
        'longitude_deg': sim_data[SDI_LONG]*conv.RAD_TO_DEG,
        'latitude_deg': sim_data[SDI_LAT]*conv.RAD_TO_DEG,
        'localGravity_ft_s2': sim_data[SDI_GRAVITY] *conv.M_TO_FT,
        'eulerAngle_deg_Yaw':  sim_data[SDI_YAW] *conv.RAD_TO_DEG,
        'eulerAngle_deg_Pitch': sim_data[SDI_PITCH] *conv.RAD_TO_DEG,
        'eulerAngle_deg_Roll' : sim_data[SDI_ROLL] *conv.RAD_TO_DEG,
        'aero_bodyForce_lbf_X': sim_data[SDI_FX] *conv.N_TO_LBF,
        'aero_bodyForce_lbf_Y': sim_data[SDI_FY] *conv.N_TO_LBF,
        'aero_bodyForce_lbf_Z': sim_data[SDI_FZ] *conv.N_TO_LBF,
        'aero_bodyMoment_ftlbf_L': sim_data[SDI_MX] *conv.NM_TO_LBF_FT,
        'aero_bodyMoment_ftlbf_M': sim_data[SDI_MY] *conv.NM_TO_LBF_FT,
        'aero_bodyMoment_ftlbf_N': sim_data[SDI_MZ] *conv.NM_TO_LBF_FT,
        'trueAirspeed_nmi_h': sim_data[SDI_TAS]*conv.MPS_TO_KTS,
        'airDensity_slug_ft3': sim_data[SDI_AIR_DENSITY] *conv.M_TO_FT,
        'downrageDistance_m': sim_data[SDI_DOWNRANGE],
        }

        return csv_data
