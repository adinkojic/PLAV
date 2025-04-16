"""Basic Flight Control for the Sim"""

import numpy as np
from numba import float64    # import the types
from numba.experimental import jitclass
from numba import jit

spec=[
    #servo control
    ('port_control', float64),
    ('star_control', float64),
    ('top_control', float64),

    ('longitude', float64),
    ('latitude', float64),

    ('euler_orientation', float64[:]),
    ('ned_velocity', float64[:]),

    ('accel', float64[:]),
    ('gyro', float64[:]),

    ('alt', float64)

]

#@jitclass(spec)
class FlightControl(object):
    """The Flight Control Program"""
    def __init__(self):
        self.longitude = 0.0
        self.latitude = 0.0
        self.euler_orientation = 0.0
        self.ned_velocity = 0.0
        self.accel = 0.0
        self.gyro = 0.0
        self.alt = 0.0
        self.airspeed = 0.0
        self.alpha = 0.0
        self.beta = 0.0

        self.yaw_command = 0.0
        self.pitch_command = 0.0
        self.roll_command = 0.0

    def give_data(self, longitude, latitude, euler_orientation, ned_velocity, accel, gyro, \
                  alt, airspeed, alpha, beta):
        """loads real data into the simulator"""
        self.longitude = longitude
        self.latitude = latitude
        self.euler_orientation = euler_orientation
        self.ned_velocity = ned_velocity
        self.accel = accel
        self.gyro = gyro
        self.alt = alt
        self.airspeed = airspeed
        self.alpha = alpha
        self.beta = beta

    def update_control_output(self):
        """updates the control output """

        self.yaw_command = 0.0
        self.pitch_command = 0.0
        self.roll_command = 0.0

        #yaw_adjustment_factor = 1.0

        #self.port_control = -self.roll_command - self.yaw_command*yaw_adjustment_factor + self.pitch_command
        #self.star_control = -self.roll_command + self.yaw_command*yaw_adjustment_factor + self.pitch_command
        #self.top_control  = -self.roll_command + self.yaw_command

    def get_control_output(self):
        """gets the control output, does not update it"""
        return np.array([self.roll_command, self.pitch_command, self.yaw_command])
