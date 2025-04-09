"""test file"""

import numpy as np
from step_logging import SimDataLogger

bruh = SimDataLogger()

alt_lon_lat = np.array([3.0, 3.1, 3.2])
ned_velocity = np.array([4.0, 4.1, 4.2])
body_rate = np.array([2.0, 2.1, 2.3])
quat = np.array([1.0, 1.1, 1.2, 1.3])

aero_body_force = np.array([5.0, 5.1, 5.2])
aero_body_moment = np.array([6.0, 6.1, 6.2])

local_gravity = 7.0
speed_of_sound = 8.0

mach = 9.0
dynamic_pressure = 10.0
true_airspeed = 11.0

air_density = 12.0
ambient_pressure = 13.0
ambient_temperature = 14.0

state = np.concat((quat, body_rate, alt_lon_lat, ned_velocity))

bruh.load_line(state, aero_body_force, aero_body_moment, local_gravity, speed_of_sound, \
        mach, dynamic_pressure, true_airspeed, air_density, ambient_pressure, ambient_temperature)

print(bruh.data)

print(bruh.make_line())

bruh.save_line()
print(bruh.data)
bruh.save_line()
print("size", np.size(bruh.data) // bruh.data_columns)
print("vals", bruh.valid_data_size)
print(bruh.tried)
bruh.save_line()
print("size", np.size(bruh.data) // bruh.data_columns)
print("vals", bruh.valid_data_size)
print(bruh.tried)
print(bruh.new_data_attempt)
print(bruh.data)

print(bruh.valid_data_size)

data = bruh.return_data()

print(data)
