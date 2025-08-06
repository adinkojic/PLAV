"""Basic RK4 Integrator
No step estimation because it shits itself around date line"""

from numba import jit, float64
import numpy as np

@jit
def basic_rk4(func, t0, step, y0, args):
    """Basic RK4 Integrator"""

    k1 = func(t0, y0, *args)
    k2 = func(t0 + step*0.5, y0 + step*k1*0.5, *args)
    k3 = func(t0 + step*0.5, y0 + step*k2*0.5, *args)
    k4 = func(t0 + step, y0 + step*k3, *args)

    y1 = y0 + step* (k1 + 2.0*k2 +2.0*k3 + k4) / 6.0

    return t0 + step, y1
