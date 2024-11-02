import math
import numpy as np
from numba import jit, prange


earth_semi_major_axis = 6378137.0 #meters
earth_semi_minor_axis = 6356752.314245 #meters
e = 8.1819190842622e-2

#from ellipseoid height
@jit
def from_lat_long_alt(lat, long, ellipsoid_alt):
    N = earth_semi_major_axis/ \
        math.sqrt(1-e**2 * math.sin(lat*math.pi/180)**2)
    point_radius = N + ellipsoid_alt
    x = point_radius * math.cos(lat * math.pi/180) * math.cos(long * math.pi/180)
    y = point_radius * math.cos(lat * math.pi/180) * math.sin(long * math.pi/180)
    z = (N*(earth_semi_minor_axis/earth_semi_major_axis)**2 +ellipsoid_alt) * math.sin(lat * math.pi/180)

    return np.array([x,y,z])

@jit
def to_lat_long_alt(xyz):

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    e_sq = e**2                     
    eps = e_sq / (1.0 - e_sq)

    p = math.sqrt(x * x + y * y)
    q = math.atan2((z * earth_semi_major_axis), (p * earth_semi_minor_axis))

    sin_q = math.sin(q)
    cos_q = math.cos(q)

    sin_q_3 = sin_q * sin_q * sin_q
    cos_q_3 = cos_q * cos_q * cos_q

    phi = math.atan2((z + eps * earth_semi_minor_axis * sin_q_3), (p - e_sq * earth_semi_major_axis * cos_q_3))
    lam = math.atan2(y, x)

    v = earth_semi_major_axis / math.sqrt(1.0 - e_sq * math.sin(phi) * math.sin(phi))
    h   = (p / math.cos(phi)) - v

    lat = math.degrees(phi)
    lon = math.degrees(lam)

    return np.array([lat,lon,h])

#wont jit idk why
#@jit(parallel=True )
def to_lat_long_xyz_array(xyz_array):
    
    working = xyz_array.transpose() #this cannot be efficient...
    lat_long_h_result = np.zeros([xyz_array.shape[1],3])

    for i in range(0, xyz_array.shape[1]):
        lat_long_h_result[i] = to_lat_long_alt(working[i])

    result = lat_long_h_result.transpose() #another tranpose op...
    return result
        
def from_NED_xyz(xyz, orientation):
    pass

def from_NED_llh():
    pass