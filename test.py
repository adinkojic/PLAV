import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84

lat = 0
long = 0
h = 0

north = np.array([1, 0, 0])
east  = np.array([0, 1, 0])
down  = np.array([0, 0, 1])

init_ori = quat.from_euler(math.pi/2,math.pi,math.pi/2) #roll pitch yaw


or2 = wgs84.from_NED_lat_long_h(np.array([lat, long, h]))

or3 = quat.mulitply(or2, init_ori)
print(or3)
print(quat.rotateFrameQ(or3, north))
print(quat.rotateFrameQ(or3, east))
print(quat.rotateFrameQ(or3, down))
