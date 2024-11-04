import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84

lat = 90
long = 0
h = 0

north = np.array([1, 0, 0])
east  = np.array([0, 1, 0])
down  = np.array([0, 0, 1])

init_ori = quat.from_angle_axis(math.pi/2, [0, -1, 0])

or2 = wgs84.from_NED_lat_long_h(np.array([lat, long, h]), init_ori)


print(or2)
print(quat.rotateFrameQ(or2, north))
print(quat.rotateFrameQ(or2, east))
print(quat.rotateFrameQ(or2, down))