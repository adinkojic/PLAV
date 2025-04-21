"""One-line log for extracting certain aircraft properties"""

import math
import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

import quaternion_math as quat

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

    #derived
    ('alpha', float64),
    ('beta', float64),
    ('reynolds', float64),

    ('data', float64[:,:]),
    ('data_columns', int64),
    ('valid_data_size', int64)


]

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

        line = self.make_line()

        data_columns = np.size(line)

        self.data_columns = int64(np.size(line))
        self.data = np.zeros((data_columns, int64(preallocated)))
        self.valid_data_size = 0

    def load_line(self, time, state, aero_body_force, \
                    aero_body_moment, local_gravity, speed_of_sound, mach ,dynamic_pressure, \
                    true_airspeed, air_density, ambient_pressure, ambient_temperature, \
                    alpha, beta, reynolds, thrust):
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

    def make_line(self):
        """Makes a line of data"""

        rollpitchyaw = quat.to_euler(self.quat)
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
            flight_path, downrange, self.thrust
         ], 'd')
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
    
    """TODO: edit
    Vincenty's inverse formula for ellipsoidal distance on WGS‑84.
    Parameters:
      lat1, lon1 — latitude and longitude of point A in rad
      lat2, lon2 — latitude and longitude of point B in rad

    Returns:
      Distance in meters.  May fail to converge near antipodal points.
    """

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

@jit(float64(float64, float64, float64, float64))
def dist_haversine(lat1, lon1, lat2, lon2):
    """
    Standard haversine formula.
    Good to ~0.1% error everywhere.
    """

    R_WGS84 = 6371008.8

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lamb = math.radians(lon2 - lon1)

    sin_delta_phi = math.sin(delta_phi * 0.5)
    sin_delta_lamb = math.sin(delta_lamb * 0.5)
    a = sin_delta_phi * sin_delta_phi + math.cos(phi1) * math.cos(phi2) * sin_delta_lamb * sin_delta_lamb
    # guard against rounding errors:
    #a = min(1.0, max(0.0, a))
    return 2 * R_WGS84 * math.asin(math.sqrt(a))