"""One-line log for extracting certain aircraft properties"""

import math
import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

from plav.quaternion_math import to_euler

spec = [

    ('time', float64),

    #positionals
    ('lon_lat_alt', float64[:]),
    ('ned_velocity', float64[:]),

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

    #derived
    ('alpha', float64),
    ('beta', float64),
    ('reynolds', float64),

    ('data', float64[:,:]),
    ('data_columns', int64),
    ('valid_data_size', int64)


]

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

SDI_LINE_SIZE = 41 #total size of a line

@jitclass(spec)
class SimDataLogger(object):
    """Jitted Logger Object to Run in the Function as an arg"""

    def __init__(self, preallocated = 1):
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

        self.control_deflection = np.zeros(4,'d')

        line = self.make_line()

        data_columns = np.size(line)

        self.data_columns = int64(np.size(line))
        self.data = np.zeros((data_columns, int64(preallocated)))
        self.valid_data_size = 0

    def get_lastest(self):
        """Returns the last line of data"""
        if self.valid_data_size == 0:
            return None
        else:
            return self.data[:, self.valid_data_size - 1]

    def load_line(self, time, state, aero_body_force, \
                    aero_body_moment, local_gravity, speed_of_sound, mach ,dynamic_pressure, \
                    true_airspeed, air_density, ambient_pressure, ambient_temperature, \
                    alpha, beta, reynolds, thrust, control_deflection):
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

    def make_line(self):
        """Makes a line of data"""

        rollpitchyaw = to_euler(self.quat)
        flight_path = calculate_flight_path_angle(self.ned_velocity)

        if self.valid_data_size != 0:
            inital_lat = self.data[9][0]
            inital_lon = self.data[8][0]
            downrange = dist_vincenty(self.lon_lat_alt[1], self.lon_lat_alt[0],inital_lat,inital_lon)
        else:
            downrange = 0.0

        line = np.array([ \
            self.time, self.quat[0], self.quat[1], self.quat[2], self.quat[3], \
            self.body_rate[0], self.body_rate[1], self.body_rate[2], \
            self.lon_lat_alt[0], self.lon_lat_alt[1], self.lon_lat_alt[2], \
            self.ned_velocity[0], self.ned_velocity[1], self.ned_velocity[2], \
            rollpitchyaw[0], rollpitchyaw[1], rollpitchyaw[2], \
            self.aero_body_force[0], self.aero_body_force[1], self.aero_body_force[2], \
            self.aero_body_moment[0], self.aero_body_moment[1], self.aero_body_moment[2], \
            self.local_gravity, self.speed_of_sound, self.mach, self.dynamic_pressure, \
            self.air_density, self.ambient_pressure, self.ambient_temperature, \
            self.true_airspeed, self.alpha, self.beta, self.reynolds, \
            flight_path, downrange, self.thrust, self.control_deflection[0], \
            self.control_deflection[1], self.control_deflection[2], self.control_deflection[3] \
         ], 'd')
        
        temp = line.copy()
        
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
        line[SDI_AILERON_CMD] = self.control_deflection[0]
        line[SDI_ELEVATOR_CMD] = self.control_deflection[1]
        line[SDI_RUDDER_CMD] = self.control_deflection[2]
        line[SDI_THRUST_CMD] = self.control_deflection[3]

        assert np.array_equal(line, temp)

        return line

    def save_line(self):
        """Saves the currently loaded data as a line"""
        line = self.make_line()
        self.append_data(line)

    def increase_size(self):
        """Double the data size if necessary"""
        self.data = np.append(self.data, self.data, axis=1) #np.pad() isn't implemented in numba lol

    def append_data(self, new_line):
        """Append a new line of data"""          
        if self.valid_data_size + 1 > np.size(self.data) // self.data_columns:
            self.increase_size()

        self.data[:, self.valid_data_size] = new_line

        self.valid_data_size = self.valid_data_size + 1

    def trim_excess(self):
        """Trim excess data"""
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
    L = lon2 - lon1

    U1 = math.atan((1 - f) * math.tan(phi1))
    U2 = math.atan((1 - f) * math.tan(phi2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lamb = L
    for _ in range(max_iter):
        sinLambda, cosLambda = math.sin(lamb), math.cos(lamb)
        sinSigma = math.hypot(cosU2 * sinLambda,
                          cosU1 * sinU2 - sinU1 * cosU2 * cosLambda)
        if sinSigma == 0:
            return 0.0  # coincident points

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)

        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cos2Alpha = 1 - sinAlpha * sinAlpha

        cos2Sigmam = (cosSigma - 2 * sinU1 * sinU2 / cos2Alpha) if cos2Alpha != 0 else 0.0

        C = f / 16 * cos2Alpha * (4 + f * (4 - 3 * cos2Alpha))

        lamb_prev = lamb
        lamb = (L + (1 - C) * f * sinAlpha *
                (sigma + C * sinSigma * (cos2Sigmam + C * cosSigma *
                 (-1 + 2 * cos2Sigmam * cos2Sigmam))))
        if abs(lamb - lamb_prev) < tol:
            break

    # handle non-convergence: proceed with best estimate anyway
    u2 = cos2Alpha * (a*a - b*b) / (b*b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175*u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47*u2)))
    deltaSigma = (B * sinSigma * (cos2Sigmam + B / 4 * (
              cosSigma * (-1 + 2 * cos2Sigmam*cos2Sigmam) -
              B / 6 * cos2Sigmam * (-3 + 4*sinSigma*sinSigma) *
              (-3 + 4*cos2Sigmam*cos2Sigmam))))
    s = b * A * (sigma - deltaSigma)

    return np.float64(s)