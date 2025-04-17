import numpy as np
from numba import jit
from scipy.interpolate import interpn

@jit
def bilinear_interp(x, y, x_grid, y_grid, z_grid):
    """Billinear interpolation with flat extrapolation
    https://en.wikipedia.org/wiki/Bilinear_interpolation"""
    if x < x_grid[0]:
        x = x_grid[0]
    if x > x_grid[-1]:
        x = x_grid[-1]
    if y < y_grid[0]:
        y = y_grid[0]
    if y > y_grid[-1]:
        y = y_grid[-1]

    x_index = np.searchsorted(x_grid, x) -1
    y_index = np.searchsorted(y_grid, y) -1

    x1 = x_grid[x_index]
    x2 = x_grid[x_index+1]
    y1 = y_grid[y_index]
    y2 = y_grid[y_index+1]

    q11 = z_grid[y_index,   x_index  ]
    q12 = z_grid[y_index+1, x_index  ]
    q21 = z_grid[y_index,   x_index+1]
    q22 = z_grid[y_index+1, x_index+1]
    
    denom = (x2-x1)*(y2-y1)
    numer = (q11*(x2-x)*(y2-y) + q12*(x2-x)*(y-y1) + q21*(x-x1)*(y2-y) + q22*(x-x1)*(y-y1))

    return numer/denom



@jit
def get_alpha_table():
    """Get Alpha for lookups"""
    alpha_table = np.array([-10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45.])
    return alpha_table

@jit
def get_el_table():
    """get elevator deflection table for lookups"""
    elevator_table = np.array([-24., -12., 0., 12., 24.])
    return elevator_table

@jit
def cx_lookup(alpha, el):
    """Lookup for C_X, based on Alpha and Elevator Deflections [deg], [deg]"""
    alpha_table = get_alpha_table()
    elevator_table = get_el_table()

    table = np.array([
        [-.099,-.081,-.081,-.063,-.025,.044,.097,.113,.145,.167,.174,.166],
        [-.048,-.038,-.040,-.021,.016,.083,.127,.137,.162,.177,.179,.167],
        [-.022,-.020,-.021,-.004,.032,.094,.128,.130,.154,.161,.155,.138],
        [-.040,-.038,-.039,-.025,.006,.062,.087,.085,.100,.110,.104,.091],
        [-.083,-.073,-.076,-.072,-.046,.012,.024,.025,.043,.053,.047,.040]
    ])

    C_X = bilinear_interp(alpha, el, alpha_table, elevator_table, table)

    return C_X

print(cx_lookup(-9,-0.2))