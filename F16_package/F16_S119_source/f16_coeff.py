"""F16 File from F16_S119"""
import math

import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba import jit


spec = [
    #deflections, d prefix is normalized -1 to 1
    ('rdr', float64),
    ('ail', float64),
    ('el', float64),
]

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

@jitclass(spec)
class F16_Lookup(object):
    """Object used to lookup coefficients for F16 jet"""
    def __init__(self):
        self.rdr  = 0.0
        self.ail  = 0.0
        self.el   = 0.0

    def input_control(self, elevator, aileron, rudder):
        """Input the control deflections in degrees of the elevator, aileron, and rudder"""
        self.el = elevator
        self.ail = aileron
        self.rdr = rudder

    def get_coeff(self, alpha, beta):
        """Gets the aerodynamic coefficients of the F16
        alpha and beta are in degrees"""
        dele=self.el/25.0 
        dail=self.ail/20.0 
        drdr=self.rdr/30.0
        
        rtd = 180/math.pi

        #cxt=cxo(alpha,el); implement table lookup
        cy=-0.02*beta+0.021*dail+0.086*drdr
        #czt=czo(alpha) implement table lookup
        cz=czt*(1.-(beta/rtd)^2)-0.19*dele

        #clt=clo(alpha,beta); lookup table
        #dclda=dlda(alpha,beta) 
        #dcldr=dldr(alpha,beta)
        cl=clt + dclda*dail + dcldr*drdr;
        cmt=cmo(alpha,el);
        cnt=cno(alpha,beta);
        dcnda=dnda(alpha,beta);
        dcndr=dndr(alpha,beta);
        cn=cnt + dcnda*dail + dcndr*drdr;
        
        # Add damping derivative contributions
        #  and cg position terms.
        
        tvt=2*vt;b2v=bspan/tvt;cq2v=cbar*q/tvt;
        d=dampder(alpha);
        cx=cxt + cq2v*d(1);                     # EBJ: d(1) is cxq
        cy=cy + b2v*(d(3)*p + d(2)*r);          # EBJ: d(2) is cyr, d(3) is cyp
        cz=cz + cq2v*d(4);                      # EBJ: d(4) is czq
        cl=cl + b2v*(d(6)*p + d(5)*r);          # EBJ: d(5) is clr, d(6) is clp
        cm=cmt + cq2v*d(7);                     # EBJ: d(7) is cmq; removed moment xfer
        cn=cn + b2v*(d(9)*p + d(8)*r);          # EBJ: d(8) is cnr, d(9) is cnp; remove moment xfer

    def get_alpha_table(self):
        """Get Alpha for lookups"""
        alpha_table = np.array([-10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45.])
        return alpha_table
    
    def get_el_table(self):
        """get elevator deflection table for lookups"""
        elevator_table = np.array([-24., -12., 0., 12., 24.])
        return elevator_table

    def cx_lookup(self, alpha, el):
        """Lookup for C_X, based on Alpha and Elevator Deflections [deg], [deg]"""

        alpha_table = self.get_alpha_table()
        elevator_table = self.get_el_table()

        table = np.array([
            [-.099,-.081,-.081,-.063,-.025,.044,.097,.113,.145,.167,.174,.166],
            [-.048,-.038,-.040,-.021,.016,.083,.127,.137,.162,.177,.179,.167],
            [-.022,-.020,-.021,-.004,.032,.094,.128,.130,.154,.161,.155,.138],
            [-.040,-.038,-.039,-.025,.006,.062,.087,.085,.100,.110,.104,.091],
            [-.083,-.073,-.076,-.072,-.046,.012,.024,.025,.043,.053,.047,.040]
        ])

        C_X = bilinear_interp(alpha, el, alpha_table, elevator_table, table)
        return C_X
    
    def cz_lookup(self, alpha):
        """Lookup for C_Z, based on Alpha and Elevator Deflections [deg]"""

        alpha_table = self.get_alpha_table()

        table = np.array([.770,.241,-.100,-.416,-.731,-1.053, -1.366,-1.646,-1.917,-2.120,-2.248,-2.229])

        C_Z = np.interp(alpha, alpha_table, table)

        return C_Z
