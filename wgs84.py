import math
import numpy as np
from numba import jit


earth_semi_major_axis = 6378137.0 #meters
earth_semi_minor_axis = 6356752.314245 #meters
e = 8.1819190842622e-2

#from ellipseoid height
def wgs84_from_lat_long_alt(lat, long, ellipsoid_alt):
    N = earth_semi_major_axis/ \
        math.sqrt(1-e**2 * math.sin(lat*math.pi/180)**2)
    point_radius = N + ellipsoid_alt
    x = point_radius * math.cos(lat * math.pi/180) * math.cos(long * math.pi/180)
    y = point_radius * math.cos(lat * math.pi/180) * math.sin(long * math.pi/180)
    z = (N*(earth_semi_minor_axis/earth_semi_major_axis)**2 +ellipsoid_alt) * math.sin(lat * math.pi/180)

    return np.array([x,y,z])




tlat = 38.80293817
tlon = 255.47540411 
th = 1911.755

print(wgs84_from_lat_long_alt(tlat, tlon, th))