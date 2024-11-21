import numpy as np
from numba import jit
import math
import quaternion_math as quat
import wgs84


nose = np.array([1, 0, 0])
left = np.array([0, 1, 0])
tail = np.array([0, 0, 1])

roll = 90
pitch = 90
yaw = 90


ori = quat.from_euler(roll * math.pi/180, pitch * math.pi/180, yaw * math.pi/180)

print(ori)
print(quat.rotateFrameQ(ori, nose))
print(quat.rotateFrameQ(ori, left))
print(quat.rotateFrameQ(ori, tail))