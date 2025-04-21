import math
import time
from numba import jit, float64, int64



@jit(float64(float64, float64, float64, float64))
def dist_vincenty(lat1, lon1, lat2, lon2):
    """
    Vincenty's inverse formula for ellipsoidal distance on WGS‑84.
    Parameters:
      lat1, lon1 — latitude and longitude of point A in degrees
      lat2, lon2 — latitude and longitude of point B in degrees
      max_iter   — maximum number of iterations (default 20)
      tol        — convergence tolerance in radians (default 1e-12)
    Returns:
      Distance in meters.  May fail to converge near antipodal points.
    """

    max_iter=20
    tol=1e-12

    # WGS‑84 ellipsoid parameters
    a = 6378137.0               # semi-major axis (meters)
    f = 1 / 298.257223563       # flattening
    b = a * (1 - f)             # semi-minor axis

    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    L = math.radians(lon2 - lon1)

    U1 = math.atan((1 - f) * math.tan(φ1))
    U2 = math.atan((1 - f) * math.tan(φ2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lamb = L
    for _ in range(max_iter):
        sinλ, cosλ = math.sin(lamb), math.cos(lamb)
        sinσ = math.hypot(cosU2 * sinλ,
                          cosU1 * sinU2 - sinU1 * cosU2 * cosλ)
        if sinσ == 0:
            return 0.0  # coincident points

        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = math.atan2(sinσ, cosσ)

        sinα = cosU1 * cosU2 * sinλ / sinσ
        cos2α = 1 - sinα * sinα

        cos2σm = (cosσ - 2 * sinU1 * sinU2 / cos2α) if cos2α != 0 else 0.0

        C = f / 16 * cos2α * (4 + f * (4 - 3 * cos2α))

        lamb_prev = lamb
        lamb = (L + (1 - C) * f * sinα *
                (σ + C * sinσ * (cos2σm + C * cosσ *
                 (-1 + 2 * cos2σm * cos2σm))))
        if abs(lamb - lamb_prev) < tol:
            break

    # handle non-convergence: proceed with best estimate anyway
    u2 = cos2α * (a*a - b*b) / (b*b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175*u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47*u2)))
    deltaσ = (B * sinσ * (cos2σm + B / 4 * (
              cosσ * (-1 + 2 * cos2σm*cos2σm) -
              B / 6 * cos2σm * (-3 + 4*sinσ*sinσ) *
              (-3 + 4*cos2σm*cos2σm))))
    s = b * A * (σ - deltaσ)

    return s

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


positions = ((33.9400, 118.400, 33.9400, 118.5))

v_dist = dist_vincenty(*positions)
h_dist = dist_haversine(*positions)

v_start = time.perf_counter()
v_dist = dist_vincenty(*positions)
h_start = time.perf_counter()
h_dist = dist_haversine(*positions)
b_end = time.perf_counter()

print(v_dist)
print(h_dist)

print("vincenty time: ", h_start - v_start)
print("halversine time: ", b_end - v_start)
