"""One-line log for extracting certain aircraft properties"""

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

spec = [

    ('time', float64),

    #positionals
    ('lon_lat_alt', float64[:]),
    ('ned_velocity', float64[:]),

    #rotationals
    ('body_rate', float64[:]),
    ('quat', float64[:]),

    #forces and moments
    ('aero_body_force', float64[:]),
    ('aero_body_moment', float64[:]),

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

    ('data', float64[:,:]),
    ('data_columns', int64),
    ('valid_data_size', int64)


]

@jitclass(spec)
class SimDataLogger(object):
    """Jitted Logger Object to Run in the Function as an arg"""

    def __init__(self):
        self.time = 0.0

        self.lon_lat_alt = np.zeros(3)
        self.ned_velocity = np.zeros(3)
        self.body_rate = np.zeros(3)
        self.quat = np.zeros(4)

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

        line = self.make_line()

        data_columns = np.size(line)

        self.data_columns = int64(np.size(line))
        self.data = np.zeros((data_columns, 1))
        self.valid_data_size = 0

    def load_line(self, time, state, aero_body_force, \
                    aero_body_moment, local_gravity, speed_of_sound, mach ,dynamic_pressure, \
                    true_airspeed, air_density, ambient_pressure, ambient_temperature, alpha, beta):
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

    def make_line(self):
        """Makes a line of data"""

        line = np.array([ \
            self.time, self.quat[0], self.quat[1], self.quat[2], self.quat[3], \
            self.body_rate[0], self.body_rate[1], self.body_rate[2], \
            self.lon_lat_alt[0], self.lon_lat_alt[1], self.lon_lat_alt[2], \
            self.ned_velocity[0], self.ned_velocity[1], self.ned_velocity[2], \
            self.aero_body_force[0], self.aero_body_force[1], self.aero_body_force[2], \
            self.aero_body_moment[0], self.aero_body_moment[1], self.aero_body_moment[2], \
            self.local_gravity, self.speed_of_sound, self.mach, self.dynamic_pressure, \
            self.air_density, self.ambient_pressure, self.ambient_temperature, \
            self.true_airspeed, self.alpha, self.beta
         ])
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
